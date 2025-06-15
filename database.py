from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker


DATABASE_URL='postgresql://RAGBASED_owner:npg_mpfbTnJMB2R0@ep-polished-recipe-a8tkih10-pooler.eastus2.azure.neon.tech/RAGBASED?sslmode=require'

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False,autoflush=False,bind=engine)
Base = declarative_base()


async def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()





