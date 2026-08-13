import argparse
import sys
import urllib.error
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the local Streamlit server.")
    parser.add_argument("--url", default="http://127.0.0.1:8501")
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()

    health_url = f"{args.url.rstrip('/')}/_stcore/health"
    try:
        with urllib.request.urlopen(health_url, timeout=args.timeout) as response:
            body = response.read().decode("utf-8").strip()
            if response.status == 200 and body == "ok":
                print(f"Streamlit healthy: {args.url}")
                return 0
            print(f"Unexpected health response: HTTP {response.status} {body}")
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"Streamlit health check failed: {exc}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
