import uvicorn
from src.api import app as api_app
from src.frontend import app as streamlit_app


if __name__ == "__main__":
    uvicorn.run(api_app, host="0.0.0.0", port=8000)
    #streamlit_app()
