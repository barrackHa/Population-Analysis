#!/usr/bin/env python3
"""
Runner script for the Neural Population Analysis Panel App

Usage:
    python run_app.py [--port PORT]
    
Example:
    python run_app.py --port 5007
"""

import argparse
import sys
import subprocess

def main():
    parser = argparse.ArgumentParser(description='Run the Neural Population Analysis Panel App')
    parser.add_argument('--port', type=int, default=5007, help='Port to run the app on (default: 5007)')
    parser.add_argument('--show', action='store_true', help='Open browser automatically')
    
    args = parser.parse_args()
    
    print(f"Starting Neural Population Analysis App on port {args.port}...")
    print(f"Open your browser and go to: http://localhost:{args.port}")
    print("Press Ctrl+C to stop the server")
    
    try:
        # Run the panel app
        cmd = [
            sys.executable, '-m', 'panel', 'serve', 'neural_analysis_app.py',
            '--port', str(args.port),
            '--allow-websocket-origin=localhost:{}'.format(args.port)
        ]
        
        if args.show:
            cmd.append('--show')
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\nShutting down the app...")
    except Exception as e:
        print(f"Error running the app: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())