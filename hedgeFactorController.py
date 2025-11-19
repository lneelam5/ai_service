from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# Create a router instead of a FastAPI app
router = APIRouter(prefix="/api", tags=["hedge-factor"])

# --- Step 1. Define request model ---
class HedgeFactorRequest(BaseModel):
    sellerNumber: str = Field(..., description="Unique seller number")
    hedgeFactor: float = Field(..., ge=0, le=1, description="Hedge factor in decimal (e.g., 0.0025 = 25 bps)")

# --- Step 2. Define response model ---
class HedgeFactorResponse(BaseModel):
    status: str
    message: str
    data: HedgeFactorRequest

# --- Step 3. Define POST endpoint ---
@router.post("/update-hedge-factor", response_model=HedgeFactorResponse)
async def update_hedge_factor(request: HedgeFactorRequest):
    """
    Update hedge factor for a given seller.
    This would normally write to a database or another system.
    """

    # Example of input validation / business rule
    if not request.sellerNumber.isdigit():
        raise HTTPException(status_code=400, detail="Invalid seller number format")

    # --- TODO: Add your real DB or API call logic here ---
    print(f"Updating hedge factor for seller {request.sellerNumber} to {request.hedgeFactor}")

    # --- Step 4. Return JSON response ---
    return HedgeFactorResponse(
        status="success",
        message=f"Hedge factor updated for seller {request.sellerNumber}",
        data=request
    )

# Note: Root endpoint moved to main.py
