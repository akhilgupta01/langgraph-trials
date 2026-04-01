from dataclasses import dataclass
from pydantic import BaseModel, Field


class Citation(BaseModel):
    document_name: str = Field(
        description="Document name in which the attribute is defined"
    )
    page: str = Field(description="Page number or page reference from the document")
    section: str = Field(description="Section or clause reference from the document")


class ReportType(BaseModel):
    name: str = Field(description="Name of the report to be submitted to the regulator")
    description: str = Field(description="Brief description of the report")
    reportFormat: str = Field(description="Format of the report such as XML, CSV, etc.")
    reportFrequency: str = Field(
        description="Declared or inferred frequency such as daily, monthly, quarterly"
    )
    reportingAttributes: list[str] = Field(
        description="List of key reporting attributes that are required to be included in the report"
    )
    jurisdiction: str = Field(
        description="Regulatory jurisdiction such as ASIC or EMIR"
    )
    assetClasses: list[str] = Field(
        description="Applicable asset classes, such as Equity, FX, Interest Rate, etc."
    )
    citations: list[Citation] = Field(
        description="Supporting citations with document name, page, and section"
    )


class RegulatoryAttribute(BaseModel):
    name: str = Field(
        description="Attribute name exactly as used in the regulatory text"
    )
    description: str = Field(description="Business meaning of the regulatory field")
    dataType: str = Field(description="Logical data type such as string, date, decimal")
    format: str = Field(description="Declared or inferred format such as ddMMyyyy")
    optionalityRules: list[str] = Field(
        description="Rules describing the situations when the attribute is mandatory or optional."
    )
    valueRules: list[str] = Field(
        description="Rules describing value constraints (e.g., enumerated values, value ranges) and in which contexts they apply (e.g. when another attribute has a certain value)"
    )

    citations: list[Citation] = Field(
        description="Supporting citations with document name, page, and section"
    )


class RegulatoryAttributes(BaseModel):
    attributes: list[RegulatoryAttribute] = Field(
        description="Regulatory reporting attributes extracted from the documents"
    )


class ReportTypes(BaseModel):
    attributes: list[ReportType] = Field(
        description="Regulatory report types extracted from the documents"
    )
