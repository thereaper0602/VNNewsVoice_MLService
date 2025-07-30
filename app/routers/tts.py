from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
import urllib.parse

from models.response import APIResponse
from schemas.tts import TTSRequest, TTSResponse, TTSDeleteByUrlRequest
from services.tts_service import ArticleTTSService
from services.cloud_service import CloudStorageService

router = APIRouter()

@router.post("/tts", response_class=Response)
async def text_to_speech_direct(request: TTSRequest):
    """Generate TTS and return WAV file directly"""
    try:
        audio_data = ArticleTTSService.generate_tts(
            request.text,
            voice_name=request.voice_name
        )
        
        if audio_data is None:
            raise HTTPException(status_code=500, detail="Failed to generate TTS audio")
        
        return Response(
            content=audio_data,
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename=speech_{request.voice_name.lower()}.wav",
                "Content-Length": str(len(audio_data))
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating TTS: {str(e)}")

@router.post("/tts/upload", response_model=APIResponse)
async def text_to_speech_with_upload(request: TTSRequest):
    """Generate TTS and upload to cloud, return URL"""
    try:
        print(f"🚀 TTS with cloud upload: {len(request.text)} chars, voice: {request.voice_name}")
        
        result = ArticleTTSService.generate_tts_with_upload(request.text, request.voice_name)
        
        if not result:
            raise HTTPException(status_code=500, detail="Failed to generate TTS or upload to cloud")
        
        tts_response = TTSResponse(
            audio_url=result["audio_url"],
            audio_size=result["audio_size"], 
            voice_name=result["voice_name"],
            text_length=result["text_length"],
            filename=result["filename"],
            upload_timestamp=result["upload_timestamp"],
            cloud_provider=result.get("cloud_provider", "cloudinary"),
            public_id=result.get("public_id")
        )
        
        return APIResponse(
            success=True,
            data=[tts_response.model_dump()],
            message="TTS generated and uploaded successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing TTS upload: {str(e)}")

@router.delete("/tts/delete/{public_id}")
async def delete_tts_audio(public_id: str):
    """Delete audio file from cloud storage"""
    try:
        decoded_public_id = urllib.parse.unquote(public_id)
        success = CloudStorageService.delete_audio(decoded_public_id)
        
        if success:
            return APIResponse(
                success=True,
                data=[{"public_id": decoded_public_id, "deleted": True}],
                message="Audio file deleted successfully"
            )
        else:
            raise HTTPException(status_code=404, detail="Audio file not found or delete failed")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting audio: {str(e)}")

@router.get("/tts/voices")
async def get_available_voices():
    """Get list of available TTS voices"""
    voices = [
        {"name": "Zephyr", "gender": "male", "language": "en"},
        {"name": "Aria", "gender": "female", "language": "en"},
        {"name": "Nova", "gender": "male", "language": "en"},
        {"name": "Luna", "gender": "female", "language": "en"}
    ]
    
    return APIResponse(
        success=True,
        data=voices,
        message=f"Found {len(voices)} available voices"
    )


# routers/tts.py - Enhanced delete endpoint

@router.delete("/tts/delete-by-url", response_model=APIResponse)
async def delete_tts_audio_by_url(request: dict):
    """Delete audio file from cloud storage using URL"""
    try:
        audio_url = request.get("audio_url", "").strip()
        
        if not audio_url:
            raise HTTPException(status_code=400, detail="Audio URL is required")
        
        # Method 1: Extract public ID and try variations
        public_id = CloudStorageService.extract_public_id_from_url(audio_url)
        
        if public_id:
            print(f"🔧 Method 1: Trying delete with extracted public_id: {public_id}")
            success = CloudStorageService.delete_audio(public_id)
            
            if success:
                return APIResponse(
                    success=True,
                    data=[{
                        "audio_url": audio_url,
                        "public_id": public_id,
                        "deleted": True,
                        "method": "extracted_public_id"
                    }],
                    message="Audio file deleted successfully"
                )
        
        # Method 2: Search by filename pattern
        filename = audio_url.split('/')[-1].split('.')[0]  # Get filename without extension
        print(f"🔧 Method 2: Trying search and delete with filename: {filename}")
        
        success = CloudStorageService.search_and_delete_audio(filename)
        
        if success:
            return APIResponse(
                success=True,
                data=[{
                    "audio_url": audio_url,
                    "public_id": filename,
                    "deleted": True,
                    "method": "search_and_delete"
                }],
                message="Audio file found and deleted successfully"
            )
        
        # If both methods fail
        raise HTTPException(status_code=404, detail="Audio file not found or delete failed")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting audio: {str(e)}")