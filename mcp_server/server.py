"""EpiAgent MCP Data Server.

A Model Context Protocol (MCP) server that exposes public health
surveillance data as tools for the EpiAgent multi-agent system.

Tools:
    - fetch_surveillance_data: Fetches epidemic data from CDC or synthetic sources
    - get_population_data: Returns population denominators
    - list_available_sources: Lists available data sources and date ranges

Transport: stdio (for local ADK integration)

Usage:
    python -m mcp_server.server
"""

from __future__ import annotations

import json
import logging
import sys

from mcp.server import Server
from mcp.server.stdio import stdio_server
import mcp.types as types

from .data_sources.synthetic import (
    INFLUENZA,
    COVID,
    MEASLES,
    generate_epidemic,
    generate_multi_wave,
)
from .data_sources.cdc_fluview import fetch_fluview

logger = logging.getLogger(__name__)

# Create the MCP server
server = Server("epiagent-data-server")

# Population database (approximate)
POPULATION_DB = {
    "nat": 330_000_000,
    "us_national": 330_000_000,
    "hhs1": 15_000_000,
    "hhs2": 25_000_000,
    "hhs3": 30_000_000,
    "hhs4": 65_000_000,
    "hhs5": 50_000_000,
    "hhs6": 40_000_000,
    "hhs7": 10_000_000,
    "hhs8": 12_000_000,
    "hhs9": 55_000_000,
    "hhs10": 8_000_000,
    "synthetic_region": 1_000_000,
}

PATHOGEN_PROFILES = {
    "influenza": INFLUENZA,
    "covid-19": COVID,
    "covid": COVID,
    "measles": MEASLES,
}


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    """List available MCP tools."""
    return [
        types.Tool(
            name="fetch_surveillance_data",
            description=(
                "Fetch epidemic surveillance data from various sources. "
                "Returns a JSON list of daily surveillance records with "
                "cases, deaths, and population data."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "enum": ["synthetic", "cdc_fluview"],
                        "description": "Data source: 'synthetic' for generated data, 'cdc_fluview' for CDC ILINet",
                    },
                    "pathogen": {
                        "type": "string",
                        "enum": ["influenza", "covid-19", "measles"],
                        "description": "Pathogen to simulate (synthetic) or filter (cdc_fluview)",
                    },
                    "region": {
                        "type": "string",
                        "description": "Region code: 'nat' for national, 'hhs1'-'hhs10', or 'synthetic_region'",
                        "default": "synthetic_region",
                    },
                    "duration_days": {
                        "type": "integer",
                        "description": "Number of days of data to generate/fetch",
                        "default": 180,
                    },
                },
                "required": ["source", "pathogen"],
            },
        ),
        types.Tool(
            name="get_population_data",
            description="Get population data for a given region.",
            inputSchema={
                "type": "object",
                "properties": {
                    "region": {
                        "type": "string",
                        "description": "Region code",
                    },
                },
                "required": ["region"],
            },
        ),
        types.Tool(
            name="list_available_sources",
            description="List all available data sources, pathogens, and regions.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Handle tool calls."""

    if name == "fetch_surveillance_data":
        return await _handle_fetch_surveillance(arguments)
    elif name == "get_population_data":
        return await _handle_get_population(arguments)
    elif name == "list_available_sources":
        return await _handle_list_sources(arguments)
    else:
        return [types.TextContent(
            type="text",
            text=json.dumps({"error": f"Unknown tool: {name}"}),
        )]


async def _handle_fetch_surveillance(args: dict) -> list[types.TextContent]:
    """Fetch surveillance data from the requested source."""
    source = args.get("source", "synthetic")
    pathogen = args.get("pathogen", "covid-19")
    region = args.get("region", "synthetic_region")
    duration = args.get("duration_days", 180)

    try:
        if source == "synthetic":
            profile = PATHOGEN_PROFILES.get(pathogen)
            if not profile:
                return [types.TextContent(
                    type="text",
                    text=json.dumps({
                        "error": f"Unknown pathogen: {pathogen}",
                        "available": list(PATHOGEN_PROFILES.keys()),
                    }),
                )]

            records = generate_epidemic(
                profile=profile,
                duration_days=duration,
                region=region,
            )
            data = [r.to_dict() for r in records]

        elif source == "cdc_fluview":
            # Convert duration to epiweek range (approximate)
            records = fetch_fluview(
                regions=region if region != "synthetic_region" else "nat",
            )
            data = [r.to_dict() for r in records]

        else:
            return [types.TextContent(
                type="text",
                text=json.dumps({"error": f"Unknown source: {source}"}),
            )]

        return [types.TextContent(
            type="text",
            text=json.dumps({
                "source": source,
                "pathogen": pathogen,
                "region": region,
                "record_count": len(data),
                "records": data,
            }),
        )]

    except Exception as e:
        logger.error("Error fetching surveillance data: %s", e)
        return [types.TextContent(
            type="text",
            text=json.dumps({"error": str(e)}),
        )]


async def _handle_get_population(args: dict) -> list[types.TextContent]:
    """Return population for a region."""
    region = args.get("region", "").lower()
    population = POPULATION_DB.get(region)

    if population is None:
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "error": f"Unknown region: {region}",
                "available_regions": list(POPULATION_DB.keys()),
            }),
        )]

    return [types.TextContent(
        type="text",
        text=json.dumps({
            "region": region,
            "population": population,
        }),
    )]


async def _handle_list_sources(args: dict) -> list[types.TextContent]:
    """List available data sources."""
    return [types.TextContent(
        type="text",
        text=json.dumps({
            "sources": {
                "synthetic": {
                    "description": "Synthetic epidemic data generated from SEIR model",
                    "pathogens": ["influenza", "covid-19", "measles"],
                    "regions": ["synthetic_region"],
                    "features": ["configurable R0", "Poisson noise", "weekend effects"],
                },
                "cdc_fluview": {
                    "description": "CDC ILINet influenza surveillance data via Delphi Epidata API",
                    "pathogens": ["influenza"],
                    "regions": ["nat", "hhs1-hhs10"],
                    "features": ["real surveillance data", "weekly resolution"],
                },
            },
            "available_regions": list(POPULATION_DB.keys()),
        }),
    )]


async def main():
    """Run the MCP server with stdio transport."""
    logger.info("Starting EpiAgent MCP Data Server...")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
