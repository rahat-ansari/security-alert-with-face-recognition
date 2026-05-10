"""
Azure MCP Server for Security Guard System

This MCP server provides tools to interact with the Deep Seek Security Guard System.
It exposes functionality for:
- Starting/stopping the security monitoring
- Getting system status
- Viewing captured events and faces
- Configuring the security system

Usage:
    python mcp_server.py

Then connect to the MCP server using an MCP client.
"""

import asyncio
import json
import os
import sys
import logging
from pathlib import Path
from typing import Any, Optional

# Add examples directory to path for importing the security guard module
EXAMPLES_DIR = Path(__file__).parent / "examples"
sys.path.insert(0, str(EXAMPLES_DIR))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SecurityMCP")

# Try to import the security guard components
try:
    # Import the Deep Seek Refinement module - handle special filename
    import importlib.util

    security_guard_path = EXAMPLES_DIR / "# Deep Seek Refinement - Improved Versio.py"
    if security_guard_path.exists():
        spec = importlib.util.spec_from_file_location("security_guard_module", str(security_guard_path))
        if spec and spec.loader:
            security_guard_module = importlib.util.module_from_spec(spec)
            sys.modules["security_guard_module"] = security_guard_module
            spec.loader.exec_module(security_guard_module)

            # Get the config class
            SecurityGuardConfig = getattr(security_guard_module, "SecurityGuardConfig", None)
            SoundManager = getattr(security_guard_module, "SoundManager", None)
            EventLogger = getattr(security_guard_module, "EventLogger", None)
            PersonRegistry = getattr(security_guard_module, "PersonRegistry", None)
            logger.info("Successfully imported security guard modules")
            SECURITY_GUARD_AVAILABLE = True
    else:
        logger.warning(f"Security guard file not found: {security_guard_path}")
        SECURITY_GUARD_AVAILABLE = False
except Exception as e:
    logger.warning(f"Could not import security guard module: {e}")
    SECURITY_GUARD_AVAILABLE = False


# Import MCP server components
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent


# ============================================================================
# MCP SERVER IMPLEMENTATION
# ============================================================================


class SecurityGuardMCP:
    """MCP Server for Security Guard System"""

    def __init__(self):
        self.server = Server("security-guard-mcp")
        self._security_system = None
        self._event_logger = None
        self._sound_manager = None
        self._setup_handlers()

    def _setup_handlers(self):
        """Set up MCP request handlers"""

        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            """List available tools"""
            return [
                Tool(
                    name="get_system_status",
                    description="Get the current status of the security guard system",
                    inputSchema={"type": "object", "properties": {}},
                ),
                Tool(
                    name="start_security_monitoring",
                    description="Start the security monitoring system with a video source",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "video_source": {
                                "type": "string",
                                "description": "Video source (0 for webcam, or path to video file)",
                            },
                            "enable_alerts": {
                                "type": "boolean",
                                "description": "Enable sound alerts for unknown persons",
                                "default": True,
                            },
                            "min_face_quality": {
                                "type": "number",
                                "description": "Minimum face quality threshold (0-100)",
                                "default": 30.0,
                            },
                        },
                        "required": ["video_source"],
                    },
                ),
                Tool(
                    name="stop_security_monitoring",
                    description="Stop the security monitoring system",
                    inputSchema={"type": "object", "properties": {}},
                ),
                Tool(
                    name="get_captured_events",
                    description="Get recent security events from the log",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "limit": {"type": "number", "description": "Number of events to retrieve", "default": 10}
                        },
                    },
                ),
                Tool(
                    name="get_captured_faces",
                    description="Get list of captured face images",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "category": {
                                "type": "string",
                                "description": "Filter by category (KNOWN, UNKNOWN, or ALL)",
                                "default": "ALL",
                            }
                        },
                    },
                ),
                Tool(
                    name="configure_system",
                    description="Configure security system parameters",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "face_tolerance": {"type": "number", "description": "Face recognition tolerance (0.3-0.7)"},
                            "min_face_quality": {"type": "number", "description": "Minimum face quality threshold"},
                            "reid_similarity_threshold": {
                                "type": "number",
                                "description": "Person re-identification similarity threshold",
                            },
                            "alarm_debounce_frames": {
                                "type": "number",
                                "description": "Frames to wait before triggering alarm",
                            },
                        },
                    },
                ),
                Tool(
                    name="play_alert_sound",
                    description="Play the security alert sound",
                    inputSchema={"type": "object", "properties": {}},
                ),
                Tool(
                    name="get_system_info",
                    description="Get information about the security system capabilities",
                    inputSchema={"type": "object", "properties": {}},
                ),
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Any) -> list[TextContent]:
            """Handle tool calls"""

            if name == "get_system_status":
                return await self._get_system_status()
            elif name == "start_security_monitoring":
                return await self._start_security_monitoring(arguments)
            elif name == "stop_security_monitoring":
                return await self._stop_security_monitoring()
            elif name == "get_captured_events":
                return await self._get_captured_events(arguments)
            elif name == "get_captured_faces":
                return await self._get_captured_faces(arguments)
            elif name == "configure_system":
                return await self._configure_system(arguments)
            elif name == "play_alert_sound":
                return await self._play_alert_sound()
            elif name == "get_system_info":
                return await self._get_system_info()
            else:
                return [TextContent(type="text", text=f"Unknown tool: {name}")]

    async def _get_system_status(self) -> list[TextContent]:
        """Get system status"""
        if not SECURITY_GUARD_AVAILABLE:
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {"status": "error", "message": "Security guard module not available", "running": False}
                    ),
                )
            ]

        status = {
            "status": "ready",
            "running": self._security_system is not None,
            "security_guard_available": SECURITY_GUARD_AVAILABLE,
            "sound_loaded": self._sound_manager.is_loaded() if self._sound_manager else False,
        }

        return [TextContent(type="text", text=json.dumps(status, indent=2))]

    async def _start_security_monitoring(self, args: dict) -> list[TextContent]:
        """Start security monitoring"""
        if not SECURITY_GUARD_AVAILABLE:
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "status": "error",
                            "message": "Security guard module not available. Please ensure all dependencies are installed.",
                        }
                    ),
                )
            ]

        video_source = args.get("video_source", "0")
        enable_alerts = args.get("enable_alerts", True)
        min_face_quality = args.get("min_face_quality", 30.0)

        try:
            config = SecurityGuardConfig()
            config.min_face_quality = min_face_quality
            config.capture_faces = True

            self._sound_manager = SoundManager()
            self._event_logger = EventLogger()

            result = {
                "status": "starting",
                "message": f"Starting security monitoring with video source: {video_source}",
                "video_source": video_source,
                "enable_alerts": enable_alerts,
                "min_face_quality": min_face_quality,
                "note": "Use the security guard script directly for full video processing",
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        except Exception as e:
            logger.error(f"Error starting security monitoring: {e}")
            return [TextContent(type="text", text=json.dumps({"status": "error", "message": str(e)}))]

    async def _stop_security_monitoring(self) -> list[TextContent]:
        """Stop security monitoring"""
        if self._security_system is None:
            return [TextContent(type="text", text=json.dumps({"status": "not_running"}))]

        try:
            self._security_system = None
            if self._sound_manager:
                self._sound_manager.stop_alarm()

            return [TextContent(type="text", text=json.dumps({"status": "stopped"}))]
        except Exception as e:
            return [TextContent(type="text", text=json.dumps({"status": "error", "message": str(e)}))]

    async def _get_captured_events(self, args: dict) -> list[TextContent]:
        """Get captured events"""
        limit = args.get("limit", 10)

        if not self._event_logger:
            return [TextContent(type="text", text=json.dumps({"events": [], "message": "No events logged yet"}))]

        try:
            log_file = self._event_logger.log_file
            events = []

            if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    lines = f.readlines()
                    for line in lines[-limit:]:
                        try:
                            events.append(json.loads(line.strip()))
                        except:
                            pass

            return [TextContent(type="text", text=json.dumps(events, indent=2))]

        except Exception as e:
            return [TextContent(type="text", text=json.dumps({"status": "error", "message": str(e)}))]

    async def _get_captured_faces(self, args: dict) -> list[TextContent]:
        """Get captured faces"""
        category = args.get("category", "ALL")

        try:
            base_dir = Path(__file__).parent.parent
            captured_dir = base_dir / "captured_faces"

            if not captured_dir.exists():
                return [
                    TextContent(type="text", text=json.dumps({"faces": [], "message": "No captured faces directory"}))
                ]

            faces = []
            for subdir in ["known", "unknown"]:
                if category != "ALL" and subdir != category.lower():
                    continue

                subdir_path = captured_dir / subdir
                if subdir_path.exists():
                    for img_file in subdir_path.glob("*"):
                        if img_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                            faces.append({"path": str(img_file), "name": img_file.stem, "category": subdir.upper()})

            return [TextContent(type="text", text=json.dumps(faces, indent=2))]

        except Exception as e:
            return [TextContent(type="text", text=json.dumps({"status": "error", "message": str(e)}))]

    async def _configure_system(self, args: dict) -> list[TextContent]:
        """Configure system parameters"""
        if not SECURITY_GUARD_AVAILABLE:
            return [
                TextContent(
                    type="text", text=json.dumps({"status": "error", "message": "Security guard not available"})
                )
            ]

        try:
            config = SecurityGuardConfig()

            if "face_tolerance" in args:
                config.face_tolerance = args["face_tolerance"]
            if "min_face_quality" in args:
                config.min_face_quality = args["min_face_quality"]
            if "reid_similarity_threshold" in args:
                config.reid_similarity_threshold = args["reid_similarity_threshold"]
            if "alarm_debounce_frames" in args:
                config.alarm_debounce_frames = args["alarm_debounce_frames"]

            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "status": "configured",
                            "config": {
                                "face_tolerance": config.face_tolerance,
                                "min_face_quality": config.min_face_quality,
                                "reid_similarity_threshold": config.reid_similarity_threshold,
                                "alarm_debounce_frames": config.alarm_debounce_frames,
                            },
                        },
                        indent=2,
                    ),
                )
            ]

        except Exception as e:
            return [TextContent(type="text", text=json.dumps({"status": "error", "message": str(e)}))]

    async def _play_alert_sound(self) -> list[TextContent]:
        """Play alert sound"""
        try:
            if not self._sound_manager:
                self._sound_manager = SoundManager()

            if self._sound_manager.play_alarm():
                return [
                    TextContent(type="text", text=json.dumps({"status": "playing", "message": "Alert sound playing"}))
                ]
            else:
                return [
                    TextContent(
                        type="text", text=json.dumps({"status": "error", "message": "Failed to play alert sound"})
                    )
                ]

        except Exception as e:
            return [TextContent(type="text", text=json.dumps({"status": "error", "message": str(e)}))]

    async def _get_system_info(self) -> list[TextContent]:
        """Get system information"""
        info = {
            "name": "Security Guard MCP Server",
            "version": "1.0.0",
            "description": "MCP Server for Security Alert with Face Recognition",
            "security_guard_available": SECURITY_GUARD_AVAILABLE,
            "capabilities": [
                "Person Detection via YOLO",
                "Face Detection via InsightFace",
                "Face Recognition (known vs unknown)",
                "Face Attributes (age, gender, emotion)",
                "Person Re-Identification",
                "Liveness Detection",
                "Quality Assessment",
                "Alarm System",
            ],
            "tools": [
                "get_system_status",
                "start_security_monitoring",
                "stop_security_monitoring",
                "get_captured_events",
                "get_captured_faces",
                "configure_system",
                "play_alert_sound",
                "get_system_info",
            ],
        }

        return [TextContent(type="text", text=json.dumps(info, indent=2))]


# ============================================================================
# MAIN
# ============================================================================


async def main():
    """Main entry point"""
    logger.info("Starting Security Guard MCP Server...")

    # Create server instance
    app = SecurityGuardMCP()
    server = app.server

    # Run using stdio_server context manager properly (no args = uses system stdin/stdout)
    async with stdio_server() as (read_stream, write_stream):
        logger.info("MCP Server running on stdio...")
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        import traceback

        logger.error(f"Server error: {e}")
        traceback.print_exc()
