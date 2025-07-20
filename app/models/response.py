from pydantic import BaseModel, HttpUrl, Field, field_validator
from typing import Optional, List, Union
from datetime import datetime
import pytz

class APIResponse(BaseModel):
    success: bool
    data: Optional[List[dict]] = None
    message: Optional[str] = None