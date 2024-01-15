from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Text,Integer, JSON, DateTime
from sqlalchemy import func

Base = declarative_base()



db_config = {
    "DATABASE_SCHEMA": "stocks"
}

db_engine =


class Company(Base):
    __tablename__ = "data_generation_result"
    __table_args__ = {"schema": db_config["DATABASE_SCHEMA"]}
    company_name : str
    ticker: str
    description: str
    industry: str

class FinIndicators(Base):
    __tablename__ = "data_generation_result"
    __table_args__ = {"schema": db_config["DATABASE_SCHEMA"]}
    ticker_name =  Column(Text)
    date = Column(DateTime, default=func.now())
    indicators =  Column(JSON)

Base.metadata.create_all(db_engine, Base.metadata.tables.values(),checkfirst=True)