"""
Functions for interacting with SQL Server.
"""

import os
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import List, Optional

try:
    import pytds
    PYTDS_AVAILABLE = True
except ImportError as e:
    pytds = None
    PYTDS_AVAILABLE = False
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"pytds not available: {e}. SQL Server functionality will be disabled.")

from dotenv import load_dotenv
from fastapi import APIRouter, Response, encoders
from pydantic import BaseModel
from ddtrace.trace import Pin
from ddtrace import tracer
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Load dotenv in the base root refers to application_top
APP_ROOT = os.path.join(os.path.dirname(__file__), '..')
dotenv_path = os.path.join(APP_ROOT, '.env')
load_dotenv(dotenv_path)

# Prep our environment variables / upload .env to Railway.app
SQLSERVER_ENABLED = os.getenv('SQLSERVER_ENABLED', 'true').lower() == 'true'
SQLSERVER_HOST = os.getenv('SQLSERVERHOST')
SQLSERVER_PORT = int(os.getenv('SQLSERVERPORT', '1433'))
SQLSERVER_USER = os.getenv('SQLSERVERUSER')
SQLSERVER_PW = os.getenv('SQLSERVERPW')
SQLSERVER_DB = os.getenv('SQLSERVERDB')

# Connection parameters for pytds (pure Python)
CONNECTION_PARAMS = {
    'server': SQLSERVER_HOST,
    'port': SQLSERVER_PORT,
    'database': SQLSERVER_DB,
    'user': SQLSERVER_USER,
    'password': SQLSERVER_PW,
    'timeout': 30,
    'login_timeout': 30,
    # Add connection resilience options
    'autocommit': False,  # Explicit transaction control
    'readonly': False,
    # Note: tds_version auto-negotiated by default (best practice)
}

# Thread pool for async operations
executor = ThreadPoolExecutor(max_workers=10)

# Global connection (will be managed similar to postgres.py pattern)
conn = None

def get_connection():
    """Get or create SQL Server connection with improved error handling."""
    if not SQLSERVER_ENABLED:
        logger.info("SQL Server is disabled via SQLSERVER_ENABLED=false")
        return None
    
    if not PYTDS_AVAILABLE:
        logger.error("pytds is not available. SQL Server connection cannot be established.")
        return None
    
    # Check if SQL Server is configured
    if not SQLSERVER_HOST or not SQLSERVER_USER:
        logger.warning("SQL Server configuration not found. Skipping SQL Server connection.")
        return None
        
    global conn
    try:
        # More robust connection validation - check for None, closed state, and test with a simple query
        needs_reconnect = False
        
        if conn is None:
            needs_reconnect = True
            logger.info("SQL Server connection is None, establishing new connection")
        else:
            # Check if connection is still alive by testing it
            try:
                # Test connection with a simple query
                test_cursor = conn.cursor()
                test_cursor.execute("SELECT 1")
                test_cursor.fetchone()
                test_cursor.close()
            except Exception as test_error:
                logger.warning(f"SQL Server connection test failed: {test_error}. Reconnecting...")
                needs_reconnect = True
                try:
                    conn.close()
                except:
                    pass  # Ignore errors when closing potentially broken connection
                conn = None
        
        if needs_reconnect:
            logger.info(f"Attempting to connect to SQL Server: server={SQLSERVER_HOST}:{SQLSERVER_PORT} database={SQLSERVER_DB} user={SQLSERVER_USER}")
            conn = pytds.connect(**CONNECTION_PARAMS)
            # Configure the connection with the proper service name for Database Monitoring
            Pin.override(conn, service="sqlserver")
            logger.info(f"Successfully connected to SQL Server database: {SQLSERVER_DB} at {SQLSERVER_HOST}:{SQLSERVER_PORT}")
            
            # Ensure the images table exists
            ensure_table_exists()
            
        return conn
    except Exception as e:
        logger.error(f"Error connecting to SQL Server: {e}", exc_info=True)
        conn = None
        return None


def ensure_table_exists():
    """Ensure the images table exists with proper schema."""
    global conn
    if conn is None:
        return
        
    cursor = conn.cursor()
    try:
        # Check if table exists
        cursor.execute("""
            SELECT COUNT(*) 
            FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_NAME = 'images'
        """)
        table_exists = cursor.fetchone()[0] > 0
        
        if not table_exists:
            logger.info("Creating images table in SQL Server...")
            
            # Create table with schema matching PostgreSQL
            create_table_sql = """
            CREATE TABLE images (
                id INT IDENTITY(1,1) PRIMARY KEY,
                name NVARCHAR(255) NOT NULL,
                width INT NULL,
                height INT NULL,
                url NVARCHAR(500) NULL,
                url_resize NVARCHAR(500) NULL,
                date_added DATE NULL DEFAULT GETDATE(),
                date_identified DATE NULL,
                ai_labels NVARCHAR(MAX) NULL,
                ai_text NVARCHAR(MAX) NULL
            )
            """
            
            cursor.execute(create_table_sql)
            
            # Create indexes
            cursor.execute("CREATE INDEX IX_images_name ON images(name)")
            cursor.execute("CREATE INDEX IX_images_date_added ON images(date_added DESC)")
            cursor.execute("CREATE INDEX IX_images_date_identified ON images(date_identified DESC)")
            
            conn.commit()
            logger.info("Successfully created images table and indexes in SQL Server")
        else:
            logger.debug("Images table already exists in SQL Server")
            
    except Exception as e:
        logger.error(f"Error ensuring table exists: {e}", exc_info=True)
        conn.rollback()
    finally:
        cursor.close()

# Initialize connection on module load
if PYTDS_AVAILABLE:
    try:
        get_connection()
    except Exception as e:
        logger.error(f"Failed to initialize SQL Server connection: {e}")
else:
    logger.warning("pytds not available - SQL Server functionality disabled")

# Create a new router for SQL Server Routes
router_sqlserver = APIRouter(tags=["SQL Server"])

class ImageModel(BaseModel):
    """Pydantic model for image data stored in SQL Server.
    
    Attributes:
        id (int): Unique identifier for the image.
        name (str): Original filename of the image.
        width (Optional[int]): Image width in pixels.
        height (Optional[int]): Image height in pixels.
        url (Optional[str]): S3 URL of the original image.
        url_resize (Optional[str]): S3 URL of resized version.
        date_added (Optional[date]): Date image was added to database.
        date_identified (Optional[date]): Date AI analysis was performed.
        ai_labels (Optional[list]): AI-detected labels from image analysis.
        ai_text (Optional[list]): AI-extracted text from image.
    """
    id: int
    name: str
    width: Optional[int]
    height: Optional[int]
    url: Optional[str]
    url_resize: Optional[str]
    date_added: Optional[date]
    date_identified: Optional[date]
    ai_labels: Optional[list]
    ai_text: Optional[list]


async def execute_query_async(query: str, params: tuple = None):
    """Execute SELECT query asynchronously using thread pool with retry logic."""
    def _execute():
        max_retries = 2
        retry_count = 0
        
        while retry_count <= max_retries:
            connection = get_connection()
            if connection is None:
                raise Exception("Failed to get SQL Server connection")
            
            # Create database span for DBM correlation
            with tracer.trace("db.query", service="sqlserver", resource=query[:100]) as span:
                # Core DBM tags for correlation (following Datadog DBM standards)
                span.set_tag("@peer.db.system", "sqlserver")
                span.set_tag("@peer.db.name", SQLSERVER_DB)
                span.set_tag("@peer.hostname", SQLSERVER_HOST)
                span.set_tag("@peer.port", SQLSERVER_PORT)
                
                # Additional database tags for better correlation
                span.set_tag("db.statement", query)
                span.set_tag("db.host", SQLSERVER_HOST)
                span.set_tag("db.port", SQLSERVER_PORT)
                span.set_tag("db.type", "sqlserver")
                span.set_tag("db.instance", SQLSERVER_DB)
                span.set_tag("db.user", SQLSERVER_USER)
                
                # Add span kind for better DBM correlation
                span.set_tag("span.kind", "client")
                span.set_tag("component", "sqlserver")
                span.set_tag("retry.attempt", retry_count)
                
                cursor = connection.cursor()
                try:
                    cursor.execute(query, params or ())
                    result = cursor.fetchall()
                    span.set_tag("db.rows_affected", len(result) if result else 0)
                    return result
                except Exception as e:
                    span.set_traceback()
                    span.set_tag("error", True)
                    span.set_tag("error.message", str(e))
                    
                    # Check if this is a connection-related error that we can retry
                    error_str = str(e).lower()
                    is_connection_error = any(err in error_str for err in [
                        'broken pipe', 'connection', 'timeout', 'network', 'socket'
                    ])
                    
                    if is_connection_error and retry_count < max_retries:
                        logger.warning(f"SQL Server query failed with connection error (attempt {retry_count + 1}/{max_retries + 1}): {e}")
                        # Force reconnection on next attempt
                        global conn
                        conn = None
                        retry_count += 1
                        continue
                    else:
                        raise e
                finally:
                    cursor.close()
    
    return await asyncio.get_event_loop().run_in_executor(executor, _execute)


async def execute_non_query_async(query: str, params: tuple = None):
    """Execute INSERT/UPDATE/DELETE query asynchronously using thread pool with retry logic."""
    def _execute():
        max_retries = 2
        retry_count = 0
        
        while retry_count <= max_retries:
            connection = get_connection()
            if connection is None:
                raise Exception("Failed to get SQL Server connection")
            
            # Create database span for DBM correlation
            with tracer.trace("db.query", service="sqlserver", resource=query[:100]) as span:
                # Core DBM tags for correlation (following Datadog DBM standards)
                span.set_tag("@peer.db.system", "sqlserver")
                span.set_tag("@peer.db.name", SQLSERVER_DB)
                span.set_tag("@peer.hostname", SQLSERVER_HOST)
                span.set_tag("@peer.port", SQLSERVER_PORT)
                
                # Additional database tags for better correlation
                span.set_tag("db.statement", query)
                span.set_tag("db.host", SQLSERVER_HOST)
                span.set_tag("db.port", SQLSERVER_PORT)
                span.set_tag("db.type", "sqlserver")
                span.set_tag("db.instance", SQLSERVER_DB)
                span.set_tag("db.user", SQLSERVER_USER)
                
                # Add span kind for better DBM correlation
                span.set_tag("span.kind", "client")
                span.set_tag("component", "sqlserver")
                span.set_tag("retry.attempt", retry_count)
                
                cursor = connection.cursor()
                try:
                    logger.info(f"SQL Server Execute Debug - About to execute query with {len(params or ())} parameters")
                    logger.info(f"SQL Server Execute Debug - Query: {query}")
                    logger.info(f"SQL Server Execute Debug - Params: {params}")
                    
                    cursor.execute(query, params or ())
                    connection.commit()
                    rowcount = cursor.rowcount
                    
                    span.set_tag("db.rows_affected", rowcount)
                    logger.info(f"SQL Server Execute Debug - Query executed successfully, {rowcount} rows affected")
                    return rowcount
                except Exception as e:
                    span.set_traceback()
                    span.set_tag("error", True)
                    span.set_tag("error.message", str(e))
                    logger.error(f"SQL Server Execute Debug - Error during execution: {str(e)}")
                    
                    try:
                        connection.rollback()
                    except:
                        pass  # Connection might be broken, ignore rollback errors
                    
                    # Check if this is a connection-related error that we can retry
                    error_str = str(e).lower()
                    is_connection_error = any(err in error_str for err in [
                        'broken pipe', 'connection', 'timeout', 'network', 'socket'
                    ])
                    
                    if is_connection_error and retry_count < max_retries:
                        logger.warning(f"SQL Server non-query failed with connection error (attempt {retry_count + 1}/{max_retries + 1}): {e}")
                        # Force reconnection on next attempt
                        global conn
                        conn = None
                        retry_count += 1
                        continue
                    else:
                        raise e
                finally:
                    cursor.close()
    
    return await asyncio.get_event_loop().run_in_executor(executor, _execute)


@router_sqlserver.get("/get-image-sqlserver/{id}", response_model=ImageModel, response_model_exclude_unset=True)
@tracer.wrap(service="sqlserver", resource="sqlserver.get_image")
async def get_image_sqlserver(id: int):
    """
    Fetch a single image from the SQL Server database.

    Args:
        id (int): The ID of the image to fetch.

    Returns:
        ImageModel: The image data as an ImageModel instance.
    """
    try:
        query = "SELECT id, name, width, height, url, url_resize, date_added, date_identified, ai_labels, ai_text FROM images WHERE id = %s"
        result = await execute_query_async(query, (id,))
        
        if not result:
            return {"error": "Image not found"}
            
        row = result[0]
        logger.debug(f"Fetched Image SQL Server: {row[1]}")
        
        # Parse JSON fields
        ai_labels = json.loads(row[8]) if row[8] else []
        ai_text = json.loads(row[9]) if row[9] else []
        
        item = ImageModel(
            id=row[0], 
            name=row[1], 
            width=row[2], 
            height=row[3], 
            url=row[4],
            url_resize=row[5], 
            date_added=row[6], 
            date_identified=row[7], 
            ai_labels=ai_labels, 
            ai_text=ai_text
        )
        return item.model_dump()
        
    except Exception as err:
        logger.error(f"Error in get_image_sqlserver: {err}", exc_info=True)
        return {"error": str(err)}


@tracer.wrap(service="sqlserver", resource="sqlserver.get_all_images")
async def get_all_images_sqlserver(response_model=List[ImageModel]):
    """
    Fetch all images from the SQL Server database.

    Returns:
        List[ImageModel]: A list of images as ImageModel instances.
    """
    formatted_photos = []
    try:
        query = "SELECT id, name, width, height, url, url_resize, date_added, date_identified, ai_labels, ai_text FROM images ORDER BY id DESC"
        result = await execute_query_async(query)
        
        for row in result:
            # Parse JSON fields
            ai_labels = json.loads(row[8]) if row[8] else []
            ai_text = json.loads(row[9]) if row[9] else []
            
            formatted_photos.append(
                ImageModel(
                    id=row[0], 
                    name=row[1], 
                    width=row[2], 
                    height=row[3], 
                    url=row[4],
                    url_resize=row[5], 
                    date_added=row[6], 
                    date_identified=row[7], 
                    ai_labels=ai_labels, 
                    ai_text=ai_text
                )
            )
            
    except Exception as err:
        logger.error(f"Error in get_all_images_sqlserver: {err}", exc_info=True)
    
    # Add span tags for monitoring
    span = tracer.current_span()
    if span:
        span.set_tag("images.count", len(formatted_photos))
    
    return formatted_photos


@tracer.wrap(service="sqlserver", resource="sqlserver.add_image")
async def add_image_sqlserver(name: str, url: str, ai_labels: list, ai_text: list):
    """
    Add an image and its metadata to the SQL Server database.

    Args:
        name (str): The name of the image.
        url (str): The S3 URL of the image.
        ai_labels (list): Labels identified by Amazon Rekognition.
        ai_text (list): Text identified by Amazon Rekognition.
    """
    try:
        # Ensure we have valid lists for JSON conversion
        if not isinstance(ai_labels, list):
            ai_labels = []
        if not isinstance(ai_text, list):
            ai_text = []
            
        # Ensure each element is a string
        ai_labels = [str(label) for label in ai_labels]
        ai_text = [str(text) for text in ai_text]
        
        logger.debug(f"Adding image to SQL Server - AI Labels: {ai_labels}")
        logger.debug(f"Adding image to SQL Server - AI Text: {ai_text}")
        
        # Convert Python lists to JSON strings
        ai_labels_json = json.dumps(ai_labels)
        ai_text_json = json.dumps(ai_text)
        
        # Debug logging to track exact parameters
        logger.info(f"SQL Server Insert Debug - Name: {name}")
        logger.info(f"SQL Server Insert Debug - URL: {url}")
        logger.info(f"SQL Server Insert Debug - AI Labels JSON: {ai_labels_json}")
        logger.info(f"SQL Server Insert Debug - AI Text JSON: {ai_text_json}")
        
        # Insert with all required columns, using NULL for optional fields
        # This matches the PostgreSQL table structure
        # Note: pytds uses %s placeholders, not ? placeholders
        query = """INSERT INTO images 
                   (name, width, height, url, url_resize, date_added, date_identified, ai_labels, ai_text) 
                   VALUES (%s, %s, %s, %s, %s, GETDATE(), %s, %s, %s)"""
        params = (name, None, None, url, None, None, ai_labels_json, ai_text_json)
        
        logger.info(f"SQL Server Insert Debug - Query: {query}")
        logger.info(f"SQL Server Insert Debug - Params: {params}")
        logger.info(f"SQL Server Insert Debug - Param count: {len(params)}")
        
        # Add span tags for monitoring
        span = tracer.current_span()
        if span:
            span.set_tag("image.name", name)
            span.set_tag("image.url", url)
            span.set_tag("image.labels_count", len(ai_labels))
            span.set_tag("image.text_count", len(ai_text))
        
        await execute_non_query_async(query, params)
        return {"message": f"Image {name} added successfully"}
        
    except Exception as err:
        logger.error(f"Error in add_image_sqlserver: {err}", exc_info=True)
        return {"error": str(err)}


@tracer.wrap(service="sqlserver", resource="sqlserver.delete_image")
async def delete_image_sqlserver(id: int):
    """
    Delete an image from the SQL Server database.

    Args:
        id (int): The ID of the image to delete.
    """
    try:
        query = "DELETE FROM images WHERE id = %s"
        rows_affected = await execute_non_query_async(query, (id,))
        
        if rows_affected == 0:
            return {"message": f"No image with id {id} found to delete"}
        return {"message": f"Image with id {id} deleted successfully"}
        
    except Exception as err:
        logger.error(f"Error in delete_image_sqlserver: {err}", exc_info=True)
        return {"error": str(err)}


@tracer.wrap(service="sqlserver", resource="sqlserver.test_connection")
async def test_sqlserver_connection():
    """
    Test SQL Server connection and basic operations for debugging.
    
    Returns:
        dict: Test results with connection status and query tests
    """
    results = {
        "connection": False,
        "table_exists": False,
        "simple_insert": False,
        "parameter_test": False,
        "errors": []
    }
    
    try:
        # Test connection
        conn = get_connection()
        if conn:
            results["connection"] = True
            logger.info("SQL Server connection test: SUCCESS")
        else:
            results["errors"].append("Failed to establish connection")
            return results
            
        # Test table existence
        try:
            query = "SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = %s"
            result = await execute_query_async(query, ("images",))
            if result and result[0][0] > 0:
                results["table_exists"] = True
                logger.info("SQL Server table existence test: SUCCESS")
            else:
                results["errors"].append("Images table does not exist")
        except Exception as e:
            results["errors"].append(f"Table existence check failed: {str(e)}")
            
        # Test simple insert with minimal parameters
        try:
            test_query = "INSERT INTO images (name, ai_labels, ai_text) VALUES (%s, %s, %s)"
            test_params = ("test_image.jpg", "[]", "[]")
            await execute_non_query_async(test_query, test_params)
            results["simple_insert"] = True
            logger.info("SQL Server simple insert test: SUCCESS")
        except Exception as e:
            results["errors"].append(f"Simple insert failed: {str(e)}")
            
        # Test complex parameter formatting
        try:
            complex_query = """INSERT INTO images 
                             (name, width, height, url, url_resize, date_added, date_identified, ai_labels, ai_text) 
                             VALUES (%s, %s, %s, %s, %s, GETDATE(), %s, %s, %s)"""
            complex_params = ("complex_test.jpg", None, None, "https://test.com/test.jpg", None, None, '["test"]', '["text"]')
            await execute_non_query_async(complex_query, complex_params)
            results["parameter_test"] = True
            logger.info("SQL Server complex parameter test: SUCCESS")
        except Exception as e:
            results["errors"].append(f"Complex parameter test failed: {str(e)}")
            
    except Exception as e:
        results["errors"].append(f"Connection test failed: {str(e)}")
        
    return results
