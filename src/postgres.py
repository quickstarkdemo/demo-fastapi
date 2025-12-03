"""
Functions for interacting with Postgres.
"""

import os
from datetime import date
from typing import List, Optional

import psycopg
from dotenv import load_dotenv
from fastapi import APIRouter, Response, encoders
from pydantic import BaseModel
try:
    # Pin lives at ddtrace.Pin in recent versions; the old ddtrace.pin module was removed
    from ddtrace import Pin
except Exception:
    class Pin:  # fallback no-op to avoid import errors when ddtrace is absent
        @staticmethod
        def override(*args, **kwargs):
            return None
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Load dotenv in the base root refers to application_top
APP_ROOT = os.path.join(os.path.dirname(__file__), '..')
dotenv_path = os.path.join(APP_ROOT, '.env')
load_dotenv(dotenv_path)

# Prep our environment variables / upload .env to Railway.app
DB = os.getenv('PGDATABASE')
HOST = os.getenv('PGHOST')
PORT = os.getenv('PGPORT')
USER = os.getenv('PGUSER')
PW = os.getenv('PGPASSWORD')

# Instantiate a Postgres connection
try:

    logger.info(f"Attempting to connect to PostgreSQL: dbname={DB} user={USER} host={HOST} port={PORT}")
    conn = psycopg.connect(
        dbname=DB, user=USER, password=PW, host=HOST, port=PORT
    )
    # Configure the connection with the proper service name for Database Monitoring
    Pin.override(conn, service="postgres")
    # Connection successful - logging handled above
    logger.info(f"Successfully connected to PostgreSQL database: {DB} at {HOST}:{PORT}")
except Exception as e:
    logger.error(f"Error connecting to PostgreSQL: {e}", exc_info=True)
    conn = None  # Initialize conn to None so later code can check if it's None

# Create a new router for Postgres Routes
router_postgres = APIRouter(tags=["PostgreSQL"])

# PostgreSQL connection parameters logged above during connection attempt


class ImageModel(BaseModel):
    """Pydantic model for image data stored in PostgreSQL.
    
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


@router_postgres.get("/get-image-postgres/{id}", response_model=ImageModel, response_model_exclude_unset=True)
async def get_image_postgres(id: int):
    """
    Fetch a single image from the Postgres database.

    Args:
        id (int): The ID of the image to fetch.

    Returns:
        ImageModel: The image data as an ImageModel instance.
    """
    global conn
    SQL = "SELECT * FROM images WHERE id = %s"
    DATA = (id,)
    cur = None
    try:
        # Check if connection is still valid
        if conn is None or conn.closed:
            # print("Connection is closed. Attempting to reconnect...")
            logger.warning("PostgreSQL connection is closed. Attempting to reconnect...")
            # Reconnect using the same credentials
            # ---> ADD THESE LINES FOR DEBUGGING <---
            # print("-" * 20)
            # print(f"DEBUG (Reconnect): Attempting connection with:")
            # print(f"DEBUG (Reconnect):   DB={DB}")
            # print(f"DEBUG (Reconnect):   HOST={HOST}")
            # print(f"DEBUG (Reconnect):   PORT={PORT}")
            # print(f"DEBUG (Reconnect):   USER={USER}")
            # print(f"DEBUG (Reconnect):   PW={'******' if PW else None}") # Don't print password directly
            # print("-" * 20)
            # ---> END DEBUGGING LINES <---
            conn = psycopg.connect(
                dbname=DB, user=USER, password=PW, host=HOST, port=PORT
            )
            # Apply Pin for DBM when reconnecting
            Pin.override(conn, service="postgres")
            logger.info(f"Successfully reconnected to PostgreSQL database: {DB} at {HOST}:{PORT}")
            
        cur = conn.cursor()
        cur.execute(SQL, DATA)
        image = cur.fetchone()  # Just fetch the specific ID we need
        if image is None:
            return {"error": "Image not found"}
        # print(f"Fetched Image Postgres: {image[1]}") # Keep specific fetch logs if desired, or remove
        logger.debug(f"Fetched Image Postgres: {image[1]}") # Changed to debug level
        item = ImageModel(id=image[0], name=image[1], width=image[2], height=image[3], url=image[4],
                          url_resize=image[5], date_added=image[6], date_identified=image[7], ai_labels=image[8], ai_text=image[9])
        return item.model_dump()
    except Exception as err:
        # print(f"Error in get_image_postgres: {err}")
        logger.error(f"Error in get_image_postgres: {err}", exc_info=True) # Add exc_info for stack trace
        return {"error": str(err)}
    finally:
        if cur is not None:
            cur.close()


async def get_all_images_postgres(response_model=List[ImageModel]):
    """
    Fetch all images from the Postgres database.

    Returns:
        List[ImageModel]: A list of images as ImageModel instances.
    """
    global conn
    formatted_photos = []  # Initialize the list outside the try block
    cur = None  # Initialize cur to None to avoid UnboundLocalError
    try:
        # Check if connection is still valid
        if conn is None or conn.closed:
            # print("Connection is closed. Attempting to reconnect...")
            logger.warning("PostgreSQL connection is closed. Attempting to reconnect...")
            # Reconnect using the same credentials
            # ---> ADD THESE LINES FOR DEBUGGING <---
            # print("-" * 20)
            # print(f"DEBUG (Reconnect): Attempting connection with:")
            # print(f"DEBUG (Reconnect):   DB={DB}")
            # print(f"DEBUG (Reconnect):   HOST={HOST}")
            # print(f"DEBUG (Reconnect):   PORT={PORT}")
            # print(f"DEBUG (Reconnect):   USER={USER}")
            # print(f"DEBUG (Reconnect):   PW={'******' if PW else None}") # Don't print password directly
            # print("-" * 20)
            # ---> END DEBUGGING LINES <---
            conn = psycopg.connect(
                dbname=DB, user=USER, password=PW, host=HOST, port=PORT
            )
            # Apply Pin for DBM when reconnecting
            Pin.override(conn, service="postgres")
            logger.info(f"Successfully reconnected to PostgreSQL database: {DB} at {HOST}:{PORT}")
            
        cur = conn.cursor()
        cur.execute("SELECT * FROM images ORDER BY id DESC")
        images = cur.fetchall()
        for image in images:
            formatted_photos.append(
                ImageModel(
                    id=image[0], name=image[1], width=image[2], height=image[3], url=image[4], url_resize=image[
                        5], date_added=image[6], date_identified=image[7], ai_labels=image[8], ai_text=image[9]
                )
            )
    except Exception as err:
        # print(f"Error in get_all_images_postgres: {err}")
        logger.error(f"Error in get_all_images_postgres: {err}", exc_info=True) # Add exc_info for stack trace
    finally:
        if cur is not None:  # Check if cursor was created before trying to close it
            cur.close()
    return formatted_photos


async def add_image_postgres(name: str, url: str, ai_labels: list, ai_text: list):
    """
    Add an image and its metadata to the Postgres database.

    Args:
        name (str): The name of the image.
        url (str): The S3 URL of the image.
        ai_labels (list): Labels identified by Amazon Rekognition.
        ai_text (list): Text identified by Amazon Rekognition.
    """
    global conn
    cur = None
    try:
        # Check if connection is still valid
        if conn is None or conn.closed:
            # print("Connection is closed. Attempting to reconnect...")
            logger.warning("PostgreSQL connection is closed. Attempting to reconnect...")
            # Reconnect using the same credentials
            # ---> ADD THESE LINES FOR DEBUGGING <---
            # print("-" * 20)
            # print(f"DEBUG (Reconnect): Attempting connection with:")
            # print(f"DEBUG (Reconnect):   DB={DB}")
            # print(f"DEBUG (Reconnect):   HOST={HOST}")
            # print(f"DEBUG (Reconnect):   PORT={PORT}")
            # print(f"DEBUG (Reconnect):   USER={USER}")
            # print(f"DEBUG (Reconnect):   PW={'******' if PW else None}") # Don't print password directly
            # print("-" * 20)
            # ---> END DEBUGGING LINES <---
            conn = psycopg.connect(
                dbname=DB, user=USER, password=PW, host=HOST, port=PORT
            )
            # Apply Pin for DBM when reconnecting
            Pin.override(conn, service="postgres")
            logger.info(f"Successfully reconnected to PostgreSQL database: {DB} at {HOST}:{PORT}")
        
        # Ensure we have valid lists for JSON conversion
        if not isinstance(ai_labels, list):
            ai_labels = []
        if not isinstance(ai_text, list):
            ai_text = []
            
        # Ensure each element is a string
        ai_labels = [str(label) for label in ai_labels]
        ai_text = [str(text) for text in ai_text]
        
        # print(f"AI Labels: {ai_labels}") # Keep specific processing logs if desired, or change level
        # print(f"AI Text: {ai_text}")
        logger.debug(f"Adding image to Postgres - AI Labels: {ai_labels}") # Changed to debug level
        logger.debug(f"Adding image to Postgres - AI Text: {ai_text}") # Changed to debug level
            
        cur = conn.cursor()
        # For jsonb columns, use JSON format, not array
        SQL = "INSERT INTO images (name, url, ai_labels, ai_text) VALUES (%s, %s, %s::jsonb, %s::jsonb)"
        
        # Convert Python lists to JSON strings that PostgreSQL can understand
        import json
        ai_labels_json = json.dumps(ai_labels)
        ai_text_json = json.dumps(ai_text)
        
        # Use JSON strings for the jsonb columns
        DATA = (name, url, ai_labels_json, ai_text_json)

        # Attempt to write the image metadata to Postgres
        cur.execute(SQL, DATA)
        conn.commit()
        return {"message": f"Image {name} added successfully"}
    except Exception as err:
        conn.rollback()
        # print(f"Error in add_image_postgres: {err}")
        logger.error(f"Error in add_image_postgres: {err}", exc_info=True) # Add exc_info for stack trace
        return {"error": str(err)}
    finally:
        if cur is not None:
            cur.close()


async def delete_image_postgres(id: int):
    """
    Delete an image from the Postgres database.

    Args:
        id (int): The ID of the image to delete.
    """
    global conn
    cur = None
    try:
        # Check if connection is still valid
        if conn is None or conn.closed:
            # print("Connection is closed. Attempting to reconnect...")
            logger.warning("PostgreSQL connection is closed. Attempting to reconnect...")
            # Reconnect using the same credentials
            # ---> ADD THESE LINES FOR DEBUGGING <---
            # print("-" * 20)
            # print(f"DEBUG (Reconnect): Attempting connection with:")
            # print(f"DEBUG (Reconnect):   DB={DB}")
            # print(f"DEBUG (Reconnect):   HOST={HOST}")
            # print(f"DEBUG (Reconnect):   PORT={PORT}")
            # print(f"DEBUG (Reconnect):   USER={USER}")
            # print(f"DEBUG (Reconnect):   PW={'******' if PW else None}") # Don't print password directly
            # print("-" * 20)
            # ---> END DEBUGGING LINES <---
            conn = psycopg.connect(
                dbname=DB, user=USER, password=PW, host=HOST, port=PORT
            )
            # Apply Pin for DBM when reconnecting
            Pin.override(conn, service="postgres")
            logger.info(f"Successfully reconnected to PostgreSQL database: {DB} at {HOST}:{PORT}")
            
        cur = conn.cursor()
        SQL = "DELETE FROM images WHERE id = %s"
        DATA = (id,)

        # Attempt to delete the image from Postgres
        cur.execute(SQL, DATA)
        rows_deleted = cur.rowcount
        conn.commit()
        
        if rows_deleted == 0:
            return {"message": f"No image with id {id} found to delete"}
        return {"message": f"Image with id {id} deleted successfully"}
    except Exception as err:
        conn.rollback()
        # print(f"Error in delete_image_postgres: {err}")
        logger.error(f"Error in delete_image_postgres: {err}", exc_info=True) # Add exc_info for stack trace
        return {"error": str(err)}
    finally:
        if cur is not None:
            cur.close()
