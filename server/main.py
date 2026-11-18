from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from middlewares.exception_handlers import catch_exception_middleware
from routes.ask_question import router as ask_router
from routes.upload_report import router as upload_report_router

app = FastAPI(title="CliniSight API", description="API for CliniSight Lab Report Explainer")

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware exception handlers
app.middleware("http")(catch_exception_middleware)

# Routers
# 1. Asking query
app.include_router(ask_router)
# 2. Lab report upload, verification & parsing
app.include_router(upload_report_router)
