"""
Functions for interacting with MongoDB.
"""

import os
import random
import urllib.parse
import logging
from typing import Optional

import pymongo
from bson import json_util
from bson.objectid import ObjectId
from dotenv import load_dotenv
from fastapi import APIRouter, Response
from pymongo import MongoClient
from pymongo.read_preferences import ReadPreference
from pymongo.server_api import ServerApi

# Set up logging
logger = logging.getLogger(__name__)

# Load dotenv in the base root refers to application_top
APP_ROOT = os.path.join(os.path.dirname(__file__), '..')
dotenv_path = os.path.join(APP_ROOT, '.env')
load_dotenv(dotenv_path)

# Global variables for lazy initialization
_client: Optional[MongoClient] = None
_db = None
_collection = None

def get_mongo_config():
    """Get MongoDB configuration from environment variables."""
    return {
        'MONGO_CONN': os.environ.get('MONGO_CONN', ''),
        'MONGO_USER': os.environ.get('MONGO_USER', ''),
        'MONGO_PW': os.environ.get('MONGO_PW', '')
    }

def is_mongo_configured():
    """Check if MongoDB is properly configured."""
    config = get_mongo_config()
    return all([config['MONGO_CONN'], config['MONGO_USER'], config['MONGO_PW']])

def get_mongo_client():
    """Lazy initialization of MongoDB client."""
    global _client
    
    if _client is not None:
        return _client
    
    if not is_mongo_configured():
        logger.warning("MongoDB not configured - some features will be unavailable")
        return None
    
    config = get_mongo_config()
    
    # Escape special characters in connection string
    mongo_user = urllib.parse.quote_plus(config['MONGO_USER'])
    mongo_pw = urllib.parse.quote_plus(config['MONGO_PW'])
    
    # Create the connection string
    uri = f"mongodb+srv://{mongo_user}:{mongo_pw}@{config['MONGO_CONN']}/?retryWrites=true&w=majority"
    
    logger.info(f"Attempting to connect to MongoDB Atlas: {config['MONGO_CONN']}")
    
    try:
        # Create a new client and connect to the server
        _client = MongoClient(
            uri,
            server_api=ServerApi('1'),
            read_preference=ReadPreference.PRIMARY_PREFERRED,
            connectTimeoutMS=30000,
            socketTimeoutMS=30000,
            serverSelectionTimeoutMS=30000
        )
        
        # Send a ping to confirm a successful connection
        _client.admin.command('ping')
        logger.info("Successfully connected to MongoDB Atlas")
        return _client
        
    except Exception as e:
        logger.error(f"Error connecting to MongoDB: {e}", exc_info=True)
        _client = None
        return None

def get_mongo_db():
    """Get MongoDB database instance."""
    global _db
    
    if _db is not None:
        return _db
    
    client = get_mongo_client()
    if client is None:
        return None
    
    _db = client.Images
    return _db

def get_mongo_collection():
    """Get MongoDB collection instance."""
    global _collection
    
    if _collection is not None:
        return _collection
    
    db = get_mongo_db()
    if db is None:
        return None
    
    _collection = db.vite_demo_images
    return _collection

# Create a new router for MongoDB Routes
router_mongo = APIRouter(tags=["MongoDB"])

@router_mongo.get(path="/get-image-mongo/{id}")
async def get_one_mongo(id: str):
    collection = get_mongo_collection()
    if collection is None:
        return {"error": "MongoDB not available"}
    
    # Fetch one document from the collection
    try:
        result = collection.find_one({"_id": ObjectId(id)})
        if result:
            result['id'] = str(result['_id'])
            del result['_id']
            return result
        else:
            return {"error": f"Document with id {id} not found"}
    except Exception as e:
        logger.error(f"Error fetching document {id}: {e}")
        return {"error": f"Failed to fetch document: {str(e)}"}

# @router_mongo.get("/get-all-images-mongo")
async def get_all_images_mongo():
    collection = get_mongo_collection()
    if collection is None:
        return {"error": "MongoDB not available"}
    
    # Get all documents from the collection
    documents = collection.find({})
    dict_cursor = [doc for doc in documents]
    for d in dict_cursor:
        d["id"] = str(d["_id"])  # swapping _id for id
        logger.debug(f"Mongo document processed: {d}")
    resp = json_util.dumps(dict_cursor, ensure_ascii=False)
    return Response(content=resp, media_type="application/json")

# @router_mongo.post("/mongo-add-image")
async def add_image_mongo(name: str, url: str, ai_labels: list, ai_text: list):
    collection = get_mongo_collection()
    if collection is None:
        return {"error": "MongoDB not available"}
    
    # Add a image data to the collection
    document = {"name": name, "url": url,
                "ai_labels": ai_labels, "ai_text": ai_text}
    result = collection.insert_one(document)
    logger.debug(f"Inserted document into MongoDB, ID: {result.inserted_id}")
    return {"message": f"Mongo added id: {result.inserted_id}"}

@router_mongo.delete(path="/delete-all-mongo/{key}")
async def delete_all_mongo(key: str):
    collection = get_mongo_collection()
    if collection is None:
        return {"error": "MongoDB not available"}
    
    # Delete all documents from the collection
    result = collection.delete_many({key: {"$exists": True}})
    return {"message": f"Mongo deleted {result.deleted_count} documents"}

# @router_mongo.delete(path="/delete-one-mongo/{id}")
async def delete_one_mongo(id: str):
    collection = get_mongo_collection()
    if collection is None:
        return {"error": "MongoDB not available"}
    
    # Delete one document from the collection
    result = collection.delete_one({"_id": ObjectId(id)})
    return {"message": f"Mongo deleted {result.deleted_count} documents"}
