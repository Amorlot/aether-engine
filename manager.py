import subprocess
import sys
import shutil
from pathlib import Path

# --- Terminal Colors ---
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

class AetherManager:
    """
    DevOps utility for managing the Aether Engine lifecycle.
    Handles Docker orchestration, cache cleaning, and log streaming.
    """
    
    def __init__(self):
        self.container_name = "api-rust"
        self.project_root = Path(__file__).parent

    def _log(self, message: str, level: str = Colors.BLUE):
        """Internal logger with color support."""
        print(f"{level}[Aether] {message}{Colors.ENDC}")

    def run_cmd(self, cmd: str) -> None:
        """Executes a shell command with error handling."""
        try:
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError:
            self._log(f"Command failed: {cmd}", Colors.FAIL)
            sys.exit(1)
        except KeyboardInterrupt:
            self._log("\nOperation cancelled by user.", Colors.WARNING)
            sys.exit(0)

    def clean(self):
        """
        Removes temporary cache files and local artifacts.
        Does NOT remove Cargo.lock to preserve dependency versions.
        """
        self._log("Cleaning local artifacts...", Colors.WARNING)
        
        # Clean specific cache directories
        cache_path = self.project_root / "api-rust" / ".fastembed_cache"
        if cache_path.exists():
            shutil.rmtree(cache_path)
            self._log(f"Removed {cache_path}", Colors.GREEN)
        
        # Optional: Prune Docker builder cache
        self.run_cmd("docker builder prune -f")
        self._log("Cleanup complete.", Colors.GREEN)

    def start(self):
        """Orchestrates the startup of all services in detached mode."""
        self._log("Starting Aether services...")
        self.run_cmd("docker compose up -d")
        self._log("Services are running.", Colors.GREEN)

    def stop(self):
        """Gracefully stops all containers."""
        self._log("Stopping services...")
        self.run_cmd("docker compose down")
        self._log("Shutdown complete.", Colors.GREEN)

    def rebuild(self):
        """
        Performs a full system reset:
        1. Stops containers
        2. Cleans cache
        3. Rebuilds the API image from scratch (no-cache)
        4. Restarts services
        """
        self.stop()
        #self.clean()
        self._log("Rebuilding API Image (No Cache)... This may take a while.", Colors.WARNING)
        self.run_cmd(f"docker compose build --no-cache {self.container_name}")
        self.start()

    def logs(self):
        """Streams real-time logs from the Rust backend."""
        self._log(f"Streaming logs for {self.container_name} (Ctrl+C to exit)...")
        self.run_cmd(f"docker compose logs -f {self.container_name}")

    def status(self):
        """Displays the current health status of the stack."""
        self.run_cmd("docker compose ps")

if __name__ == "__main__":
    manager = AetherManager()
    
    commands = {
        "start": manager.start,
        "stop": manager.stop,
        "rebuild": manager.rebuild,
        "logs": manager.logs,
        "clean": manager.clean,
        "status": manager.status
    }

    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print(f"{Colors.HEADER}Usage: python manager.py [{'|'.join(commands.keys())}]{Colors.ENDC}")
        sys.exit(1)
    
    # Execute the requested command
    commands[sys.argv[1]]()