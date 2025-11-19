# main.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest
import sys
import time

print("Starting main.py import...")

# Add request logging middleware
class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: StarletteRequest, call_next):
        start_time = time.time()
        print(f"→ Incoming request: {request.method} {request.url.path}")
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            print(f"← Response sent: {response.status_code} (took {process_time:.3f}s)")
            return response
        except Exception as e:
            process_time = time.time() - start_time
            print(f"✗ Request failed after {process_time:.3f}s: {e}")
            raise

# Lazy import to avoid blocking on startup
try:
    print("Importing add_hedge_factor...")
    from .add_hedge_factor import agent
    print("Successfully imported agent")
except ImportError:
    try:
        from add_hedge_factor import agent
        print("Successfully imported agent (absolute)")
    except Exception as e:
        print(f"Error importing agent: {e}", file=sys.stderr)
        agent = None
except Exception as e:
    print(f"Error importing agent: {e}", file=sys.stderr)
    agent = None

try:
    print("Importing hedgeFactorController...")
    from .hedgeFactorController import router
    print("Successfully imported router")
except ImportError:
    try:
        from hedgeFactorController import router
        print("Successfully imported router (absolute)")
    except Exception as e:
        print(f"Error importing router: {e}", file=sys.stderr)
        router = None
except Exception as e:
    print(f"Error importing router: {e}", file=sys.stderr)
    router = None

app = FastAPI(title="Hedge Factor API", version="1.0")

# Add logging middleware FIRST (before CORS)
app.add_middleware(LoggingMiddleware)

# Include the hedge factor router
if router:
    app.include_router(router)
    print("Router included successfully")

# Allow your frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/chat")
async def chat(request: Request):
    print("Chat request received")
    if agent is None:
        return {"error": "Agent not initialized. Check server logs."}
    
    body = await request.json()
    user_input = body.get("input", "")
    if not user_input:
        return {"error": "Missing 'input' field"}

    try:
        result = agent.invoke({"input": user_input})
        # The hedge factor agent returns api_response instead of output
        if "api_response" in result:
            return {"response": result["api_response"]}
        elif "output" in result:
            return {"response": result["output"]}
        else:
            return {"response": result}
    except Exception as e:
        print(f"Error in chat endpoint: {e}", file=sys.stderr)
        return {"error": f"Agent execution failed: {str(e)}"}

@app.get("/")
async def root():
    print("Root endpoint called!")
    try:
        response = {
            "message": "Hedge Factor API is running!",
            "status": "ok",
            "agent_loaded": agent is not None
        }
        print(f"Returning response: {response}")
        return response
    except Exception as e:
        print(f"Error in root endpoint: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

@app.get("/health")
async def health():
    """Simple health check endpoint"""
    print("Health endpoint called!")
    return {"status": "healthy", "message": "Server is running"}

@app.get("/test")
async def test():
    """Ultra simple test endpoint"""
    print("Test endpoint called!")
    return {"test": "ok"}

@app.on_event("startup")
async def startup_event():
    print("FastAPI app started successfully!")
    print(f"Agent loaded: {agent is not None}")
    print(f"Router loaded: {router is not None}")

