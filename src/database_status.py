"""
Database status checking utilities for FastAPI application.
Provides health checks and configuration validation for all database backends.
"""

import os
import logging
from typing import Dict, Any
from fastapi import APIRouter

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Database Status"])


def check_sqlserver_status() -> Dict[str, Any]:
    """Check SQL Server configuration and availability."""
    from src.sqlserver import SQLSERVER_ENABLED, SQLSERVER_HOST, SQLSERVER_PORT, SQLSERVER_DB, PYTDS_AVAILABLE, get_connection
    
    status = {
        "name": "SQL Server",
        "enabled": SQLSERVER_ENABLED,
        "available": False,
        "configured": False,
        "connection": "not_attempted",
        "details": {}
    }
    
    # Check if SQL Server is disabled
    if not SQLSERVER_ENABLED:
        status["connection"] = "disabled"
        status["details"]["message"] = "SQL Server is disabled via SQLSERVER_ENABLED=false"
        return status
    
    # Check if pytds is available
    if not PYTDS_AVAILABLE:
        status["connection"] = "error"
        status["details"]["error"] = "pytds library not available"
        status["details"]["message"] = "Install with: pip install python-tds"
        return status
    
    # Check configuration
    if not SQLSERVER_HOST or not SQLSERVER_DB:
        status["connection"] = "error"
        status["configured"] = False
        status["details"]["error"] = "Missing required configuration"
        status["details"]["required"] = ["SQLSERVERHOST", "SQLSERVERDB", "SQLSERVERUSER", "SQLSERVERPW"]
        return status
    
    status["configured"] = True
    status["details"]["host"] = SQLSERVER_HOST
    status["details"]["port"] = SQLSERVER_PORT
    status["details"]["database"] = SQLSERVER_DB
    
    # Try to connect
    try:
        conn = get_connection()
        if conn:
            status["available"] = True
            status["connection"] = "connected"
            status["details"]["message"] = "Successfully connected to SQL Server"
        else:
            status["connection"] = "failed"
            status["details"]["error"] = "Failed to establish connection"
    except Exception as e:
        status["connection"] = "error"
        status["details"]["error"] = str(e)
    
    return status


def check_postgres_status() -> Dict[str, Any]:
    """Check PostgreSQL configuration and availability."""
    status = {
        "name": "PostgreSQL",
        "enabled": True,  # PostgreSQL is always enabled if configured
        "available": False,
        "configured": False,
        "connection": "not_attempted",
        "details": {}
    }
    
    try:
        from src.postgres import PGHOST, PGPORT, PGDATABASE, get_connection
        
        if not PGHOST or not PGDATABASE:
            status["configured"] = False
            status["connection"] = "error"
            status["details"]["error"] = "Missing required configuration"
            status["details"]["required"] = ["PGHOST", "PGDATABASE", "PGUSER", "PGPASSWORD"]
            return status
        
        status["configured"] = True
        status["details"]["host"] = PGHOST
        status["details"]["port"] = PGPORT
        status["details"]["database"] = PGDATABASE
        
        # Try to connect
        conn = get_connection()
        if conn:
            status["available"] = True
            status["connection"] = "connected"
            status["details"]["message"] = "Successfully connected to PostgreSQL"
        else:
            status["connection"] = "failed"
            status["details"]["error"] = "Failed to establish connection"
    except Exception as e:
        status["connection"] = "error"
        status["details"]["error"] = str(e)
    
    return status


def check_mongodb_status() -> Dict[str, Any]:
    """Check MongoDB configuration and availability."""
    status = {
        "name": "MongoDB",
        "enabled": True,  # MongoDB is always enabled if configured
        "available": False,
        "configured": False,
        "connection": "not_attempted",
        "details": {}
    }
    
    try:
        from src.mongo import MONGO_CONN, client
        
        if not MONGO_CONN:
            status["configured"] = False
            status["connection"] = "error"
            status["details"]["error"] = "Missing required configuration"
            status["details"]["required"] = ["MONGO_CONN", "MONGO_USER", "MONGO_PW"]
            return status
        
        status["configured"] = True
        status["details"]["connection_string"] = "***configured***"  # Don't expose the actual connection string
        
        # Try to ping MongoDB
        if client:
            client.admin.command('ping')
            status["available"] = True
            status["connection"] = "connected"
            status["details"]["message"] = "Successfully connected to MongoDB"
        else:
            status["connection"] = "failed"
            status["details"]["error"] = "Client not initialized"
    except Exception as e:
        status["connection"] = "error"
        status["details"]["error"] = str(e)
    
    return status


@router.get("/database-status")
async def get_database_status():
    """
    Get the status of all database backends.
    Shows configuration, availability, and connection status.
    """
    status = {
        "databases": {
            "sqlserver": check_sqlserver_status(),
            "postgres": check_postgres_status(),
            "mongodb": check_mongodb_status()
        },
        "summary": {
            "total": 3,
            "enabled": 0,
            "configured": 0,
            "available": 0
        }
    }
    
    # Calculate summary
    for db_status in status["databases"].values():
        if db_status.get("enabled"):
            status["summary"]["enabled"] += 1
        if db_status.get("configured"):
            status["summary"]["configured"] += 1
        if db_status.get("available"):
            status["summary"]["available"] += 1
    
    return status


@router.get("/database-config")
async def get_database_config():
    """
    Get the current database configuration (without sensitive data).
    Useful for debugging connection issues.
    """
    from src.sqlserver import SQLSERVER_ENABLED, SQLSERVER_HOST, SQLSERVER_PORT, SQLSERVER_DB
    from src.postgres import PGHOST, PGPORT, PGDATABASE
    
    config = {
        "sqlserver": {
            "enabled": SQLSERVER_ENABLED,
            "host": SQLSERVER_HOST or "not_configured",
            "port": SQLSERVER_PORT,
            "database": SQLSERVER_DB or "not_configured",
            "user": "***configured***" if os.getenv('SQLSERVERUSER') else "not_configured"
        },
        "postgres": {
            "enabled": True,
            "host": PGHOST or "not_configured",
            "port": PGPORT,
            "database": PGDATABASE or "not_configured",
            "user": "***configured***" if os.getenv('PGUSER') else "not_configured"
        },
        "mongodb": {
            "enabled": True,
            "connection": "***configured***" if os.getenv('MONGO_CONN') else "not_configured"
        }
    }
    
    return config
