#!/usr/bin/env python3
"""
FocusGuard - Main Entry Point
Real-Time Procrastination Detection Application

Usage:
    python main.py start       - Full workflow (calibrate if needed, then detect)
    python main.py calibrate   - Run calibration phase only
    python main.py detect      - Run detection phase only
    python main.py server      - Start web dashboard server
    python main.py --help      - Show this help message
"""

import sys
import argparse
import logging
import signal
import atexit
from pathlib import Path

# Setup basic logging early
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

# Import project modules
try:
    from config import *
    from app_controller import FocusGuardController
    from web_server import run_server
    from activity_stream import RealTimeActivityMonitor
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.error("Please ensure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)


def print_banner():
    """Print application banner"""
    banner = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   🎯 {APP_NAME} - {APP_DESCRIPTION}                    ║
║                                                                              ║
║   Version: {APP_VERSION}                                                           ║
║   Privacy-First • Real-Time • Machine Learning Powered                      ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown"""
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def check_system_requirements():
    """Check if system requirements are met"""
    requirements_met = True
    errors = []
    
    # Check Python version
    if sys.version_info < (3, 8):
        requirements_met = False
        errors.append("Python 3.8+ is required")
    
    # Check Windows-specific requirements
    try:
        import psutil
        import win32gui
        import win32process
        from pynput import mouse, keyboard
    except ImportError as e:
        requirements_met = False
        errors.append(f"Missing Windows monitoring dependencies: {e}")
        errors.append("Please install: pip install psutil pywin32 pynput")
    
    # Check ML dependencies
    try:
        import numpy
        import pandas
        import sklearn
        import joblib
    except ImportError as e:
        requirements_met = False
        errors.append(f"Missing ML dependencies: {e}")
        errors.append("Please install: pip install numpy pandas scikit-learn joblib")
    
    # Check web server dependencies
    try:
        import flask
    except ImportError as e:
        requirements_met = False
        errors.append(f"Missing web server dependencies: {e}")
        errors.append("Please install: pip install flask")
    
    if not requirements_met:
        logger.error("System requirements not met:")
        for error in errors:
            logger.error(f"  - {error}")
        return False
    
    return True


def cmd_start(args):
    """Start full workflow (calibrate if needed, then detect)"""
    logger.info("Starting FocusGuard full workflow...")
    
    controller = FocusGuardController()
    
    try:
        # Check if model exists
        if not MODEL_FILE.exists():
            logger.info("No trained model found. Starting calibration...")
            print("\n🔧 No model found. Starting calibration phase...")
            controller.phase1_calibration()
            print("✅ Calibration completed successfully!")
        else:
            logger.info("Trained model found. Skipping calibration.")
            print("✅ Model found. Skipping calibration.")
        
        # Start detection
        print("\n🚀 Starting real-time detection...")
        duration = args.duration if hasattr(args, 'duration') and args.duration else None
        controller.phase2_detection(duration=duration)
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        print("\n⏹️  Stopped by user")
    except Exception as e:
        logger.error(f"Error in main workflow: {e}")
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0


def cmd_calibrate(args):
    """Run calibration phase only"""
    logger.info("Starting calibration phase...")
    print("\n🔧 Starting calibration phase...")
    
    controller = FocusGuardController()
    
    try:
        controller.phase1_calibration()
        print("✅ Calibration completed successfully!")
        return 0
    except KeyboardInterrupt:
        logger.info("Calibration interrupted by user")
        print("\n⏹️  Calibration stopped by user")
        return 1
    except Exception as e:
        logger.error(f"Error in calibration: {e}")
        print(f"\n❌ Calibration error: {e}")
        return 1


def cmd_detect(args):
    """Run detection phase only"""
    if not MODEL_FILE.exists():
        logger.error("No trained model found. Run calibration first.")
        print("❌ No trained model found. Please run calibration first:")
        print("   python main.py calibrate")
        return 1
    
    logger.info("Starting detection phase...")
    print("\n🚀 Starting real-time detection...")
    
    controller = FocusGuardController()
    
    try:
        duration = args.duration if hasattr(args, 'duration') and args.duration else None
        controller.phase2_detection(duration=duration)
        return 0
    except KeyboardInterrupt:
        logger.info("Detection interrupted by user")
        print("\n⏹️  Detection stopped by user")
        return 0
    except Exception as e:
        logger.error(f"Error in detection: {e}")
        print(f"\n❌ Detection error: {e}")
        return 1


def cmd_server(args):
    """Start web dashboard server"""
    logger.info("Starting web dashboard server...")
    print(f"\n🌐 Starting web dashboard server...")
    print(f"   URL: http://{WEB_SERVER_HOST}:{WEB_SERVER_PORT}")
    print(f"   Press Ctrl+C to stop")
    
    try:
        run_server()
        return 0
    except KeyboardInterrupt:
        logger.info("Web server stopped by user")
        print("\n⏹️  Web server stopped")
        return 0
    except Exception as e:
        logger.error(f"Error starting web server: {e}")
        print(f"\n❌ Web server error: {e}")
        return 1


def cmd_test(args):
    """Test real-time monitoring functionality"""
    logger.info("Testing real-time monitoring...")
    print("\n🧪 Testing real-time activity monitoring...")
    
    try:
        # Import configuration
        import config
        
        # Create monitor
        monitor = RealTimeActivityMonitor(config)
        
        print("   Starting monitoring for 10 seconds...")
        monitor.start_monitoring()
        
        import time
        start_time = time.time()
        
        while time.time() - start_time < 10:
            events = monitor.get_events(since_timestamp=time.time() - 1)
            if events:
                print(f"   📊 Captured {len(events)} events in last second")
                for event in events[-3:]:  # Show last 3 events
                    print(f"      {event.event_type.value}: {event.app_name}")
            time.sleep(1)
        
        monitor.stop_monitoring()
        
        total_events = len(monitor.get_events())
        print(f"\n✅ Test completed successfully!")
        print(f"   Total events captured: {total_events}")
        print(f"   Current state: {monitor.get_current_state()}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"\n❌ Test failed: {e}")
        return 1


def cmd_info(args):
    """Show system and configuration information"""
    print("\n📋 FocusGuard System Information")
    print("=" * 50)
    
    # Python info
    print(f"Python Version: {sys.version}")
    print(f"Platform: {sys.platform}")
    
    # Paths
    print(f"\nProject Root: {PROJECT_ROOT}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Models Directory: {MODELS_DIR}")
    print(f"Logs Directory: {LOGS_DIR}")
    
    # Model status
    print(f"\nModel Status:")
    print(f"  Baseline Model: {'✅ Exists' if MODEL_FILE.exists() else '❌ Not found'}")
    print(f"  Classifier Model: {'✅ Exists' if RANDOM_FOREST_MODEL_FILE.exists() else '❌ Not found'}")
    print(f"  Feature Scaler: {'✅ Exists' if SCALER_FILE.exists() else '❌ Not found'}")
    
    # Data status
    print(f"\nData Status:")
    print(f"  Raw Data: {'✅ Exists' if RAW_DATA_FILE.exists() else '❌ Not found'}")
    print(f"  Labeled Data: {'✅ Exists' if LABELED_DATA_FILE.exists() else '❌ Not found'}")
    print(f"  Session Log: {'✅ Exists' if SESSION_LOG.exists() else '❌ Not found'}")
    
    # Configuration
    print(f"\nConfiguration:")
    print(f"  Calibration Duration: {CALIBRATION_DURATION_SECONDS}s")
    print(f"  Detection Interval: {DETECTION_INTERVAL_SECONDS}s")
    print(f"  Detection Window: {DETECTION_WINDOW_SIZE}s")
    print(f"  Web Server: {WEB_SERVER_HOST}:{WEB_SERVER_PORT}")
    
    return 0


def create_parser():
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description=f"{APP_NAME} - {APP_DESCRIPTION}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py start              # Full workflow
  python main.py calibrate          # Calibration only
  python main.py detect             # Detection only
  python main.py server             # Web dashboard
  python main.py test               # Test monitoring
  python main.py info               # System info
        """
    )
    
    # Add subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Start command
    start_parser = subparsers.add_parser('start', help='Start full workflow')
    start_parser.add_argument('--duration', type=int, help='Detection duration in seconds')
    start_parser.set_defaults(func=cmd_start)
    
    # Calibrate command
    calibrate_parser = subparsers.add_parser('calibrate', help='Run calibration only')
    calibrate_parser.set_defaults(func=cmd_calibrate)
    
    # Detect command
    detect_parser = subparsers.add_parser('detect', help='Run detection only')
    detect_parser.add_argument('--duration', type=int, help='Detection duration in seconds')
    detect_parser.set_defaults(func=cmd_detect)
    
    # Server command
    server_parser = subparsers.add_parser('server', help='Start web dashboard')
    server_parser.set_defaults(func=cmd_server)
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test monitoring functionality')
    test_parser.set_defaults(func=cmd_test)
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show system information')
    info_parser.set_defaults(func=cmd_info)
    
    # Global options
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--quiet', action='store_true', help='Suppress non-error output')
    
    return parser


def main():
    """Main entry point"""
    # Setup signal handlers
    setup_signal_handlers()
    
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Configure logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Show banner unless quiet
    if not args.quiet:
        print_banner()
    
    # Check system requirements
    if not check_system_requirements():
        return 1
    
    # Ensure directories exist
    for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
        directory.mkdir(exist_ok=True)
    
    # Run command
    if hasattr(args, 'func'):
        try:
            return args.func(args)
        except Exception as e:
            logger.error(f"Command failed: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
            return 1
    else:
        # No command specified, show help
        parser.print_help()
        return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
