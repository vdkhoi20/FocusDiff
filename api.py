import tempfile
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from focusdiff import FocusDiff, FocusDiffConfig


app = FastAPI(title="FocusDiff API")


@lru_cache(maxsize=3)
def get_focusdiff(version: str, device: Optional[str], dtype: str, cache_dir: Optional[str], local_files_only: bool):
    config = FocusDiffConfig(
        torch_dtype=dtype,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    if device is not None:
        config.device = device
    return FocusDiff(version=version, config=config, device=device)


async def save_upload(upload: UploadFile, path: Path):
    content = await upload.read()
    if not content:
        raise HTTPException(status_code=400, detail=f"Uploaded file is empty: {upload.filename}")
    path.write_bytes(content)


def image_response(image) -> StreamingResponse:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/png")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/edit_image")
async def edit_image_api(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    prompt: str = Form(...),
    version: str = Form("sd15"),
    device: Optional[str] = Form(None),
    dtype: str = Form("float32"),
    cache_dir: Optional[str] = Form(None),
    local_files_only: bool = Form(False),
):
    if version not in {"sd15", "sd21", "sdxl"}:
        raise HTTPException(status_code=400, detail="version must be one of: sd15, sd21, sdxl")
    if dtype not in {"float32", "float16", "bfloat16"}:
        raise HTTPException(status_code=400, detail="dtype must be one of: float32, float16, bfloat16")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        image_path = tmp / (image.filename or "image.png")
        mask_path = tmp / (mask.filename or "mask.png")
        await save_upload(image, image_path)
        await save_upload(mask, mask_path)

        focusdiff = get_focusdiff(version, device, dtype, cache_dir, local_files_only)
        try:
            result = focusdiff.edit_image(image_path, mask_path, prompt)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return image_response(result)


@app.post("/erase_image")
async def erase_image_api(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    version: str = Form("sd15"),
    device: Optional[str] = Form(None),
    dtype: str = Form("float32"),
    cache_dir: Optional[str] = Form(None),
    local_files_only: bool = Form(False),
):
    if version not in {"sd15", "sd21", "sdxl"}:
        raise HTTPException(status_code=400, detail="version must be one of: sd15, sd21, sdxl")
    if dtype not in {"float32", "float16", "bfloat16"}:
        raise HTTPException(status_code=400, detail="dtype must be one of: float32, float16, bfloat16")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        image_path = tmp / (image.filename or "image.png")
        mask_path = tmp / (mask.filename or "mask.png")
        await save_upload(image, image_path)
        await save_upload(mask, mask_path)

        focusdiff = get_focusdiff(version, device, dtype, cache_dir, local_files_only)
        try:
            result = focusdiff.edit_image(image_path, mask_path, prompt="", do_erase=True)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return image_response(result)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8003)
