import os
import io
import shutil
import tempfile
import torch
import cv2 
import numpy as np
from pathlib import Path
from PIL import Image

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse

# Import the corrected image processing class
from run_pano import ImageInversionProcessor


app = FastAPI()

# Initialize the processor once for the application lifetime
# Note: This requires the OIIctrl library and model to be installed locally.
processor = ImageInversionProcessor()

# Assumed directory for source images
SOURCE_IMAGE_DIR = "./scenes_360"

# The following code block is for demonstration purposes only.
# In a real-world scenario, you would have your actual image files here.
if not os.path.exists(SOURCE_IMAGE_DIR):
    os.makedirs(SOURCE_IMAGE_DIR)
    for i in range(1, 11):
        dummy_image = Image.fromarray(np.full((500, 800, 3), 255, dtype=np.uint8))
        dummy_image.save(Path(SOURCE_IMAGE_DIR) / f"{i}.jpg")


@app.post("/edit_image")
async def edit_image_api(id_image: str = Form(...), mask: UploadFile = File(...), prompt: str = Form(...)):
    """
    API endpoint to process an image editing request.
    It receives id_image, a mask file, and a prompt, then returns the edited image.
    """
    
    # Create a temporary directory to store uploaded files and results
    temp_dir = tempfile.mkdtemp()
    
    try:
        # 1. Check and get the path of the original image
        source_image_path = Path(SOURCE_IMAGE_DIR) / f"{id_image}.png"
        if id_image=="1":
            source_image_path = Path(SOURCE_IMAGE_DIR) / f"{id_image}.jpg"
        if not source_image_path.is_file():
            raise HTTPException(status_code=404, detail=f"Image ID {id_image} not found.")

        # 2. Save the uploaded mask file to the temporary directory
        mask_file_path = Path(temp_dir) / "uploaded_mask.png"
        with open(mask_file_path, "wb") as buffer:
            shutil.copyfileobj(mask.file, buffer)

        # 3. Define the output directory for the processor
        output_dir = Path(temp_dir)

        # 4. Run the image processing pipeline
        # The run_pipeline method now handles all file I/O internally
        processor.run_pipeline(
            str(source_image_path),
            str(mask_file_path),
            prompt,
            output_dir=str(output_dir)
        )
        
        # 5. Construct the path to the final edited image
        final_image_name = f"final_{source_image_path.stem}_{prompt.replace(' ', '_')}.png"
        final_image_path = output_dir / "_panorama" / final_image_name

        if not final_image_path.is_file():
            raise HTTPException(status_code=500, detail="Failed to generate final image.")

        # 6. Read the final image and return it as a streaming response
        image_stream = open(final_image_path, "rb")
        
        return StreamingResponse(image_stream, media_type="image/png")
    
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # 7. Clean up the temporary directory and all its contents
        shutil.rmtree(temp_dir, ignore_errors=True)

@app.post("/erase_image")
async def edit_image_api(id_image: str = Form(...), mask: UploadFile = File(...), prompt: str = Form(...)):
    """
    API endpoint to process an image editing request.
    It receives id_image, a mask file, and a prompt, then returns the edited image.
    """
    
    # Create a temporary directory to store uploaded files and results
    temp_dir = tempfile.mkdtemp()
    
    try:
        # 1. Check and get the path of the original image
        source_image_path = Path(SOURCE_IMAGE_DIR) / f"{id_image}.png"
        if id_image=="1":
            source_image_path = Path(SOURCE_IMAGE_DIR) / f"{id_image}.jpg"
        if not source_image_path.is_file():
            raise HTTPException(status_code=404, detail=f"Image ID {id_image} not found.")

        # 2. Save the uploaded mask file to the temporary directory
        mask_file_path = Path(temp_dir) / "uploaded_mask.png"
        with open(mask_file_path, "wb") as buffer:
            shutil.copyfileobj(mask.file, buffer)

        # 3. Define the output directory for the processor
        output_dir = Path(temp_dir)

        # 4. Run the image processing pipeline
        # The run_pipeline method now handles all file I/O internally
        processor.run_pipeline(
            str(source_image_path),
            str(mask_file_path),
            prompt,
            output_dir=str(output_dir),
            DoErase=True
        )
        
        # 5. Construct the path to the final edited image
        final_image_name = f"final_{source_image_path.stem}_{prompt.replace(' ', '_')}.png"
        final_image_path = output_dir / "_panorama" / final_image_name

        if not final_image_path.is_file():
            raise HTTPException(status_code=500, detail="Failed to generate final image.")

        # 6. Read the final image and return it as a streaming response
        image_stream = open(final_image_path, "rb")
        
        return StreamingResponse(image_stream, media_type="image/png")
    
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # 7. Clean up the temporary directory and all its contents
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    import uvicorn
    # Run the server with Uvicorn. The API documentation is available at http://localhost:8003/docs
    uvicorn.run(app, host="0.0.0.0", port=8003)
