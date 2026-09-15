"""A redirect must not forward the acceptance runner's bearer credential."""
import importlib.util
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
import unittest
from urllib.error import HTTPError
from urllib.request import Request

SPEC = importlib.util.spec_from_file_location(
    "acceptance", Path(__file__).resolve().parents[1] / "tools/validate_miaai_update.py"
)
acceptance = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(acceptance)


class RedirectCredentialTest(unittest.TestCase):
    def test_cross_origin_redirect_never_receives_credential(self):
        received = []

        class Sink(BaseHTTPRequestHandler):
            def do_GET(self):
                received.append(self.headers.get("Authorization"))
                self.send_response(200)
                self.end_headers()

            def log_message(self, *args):
                pass

        sink = ThreadingHTTPServer(("127.0.0.1", 0), Sink)
        target = f"http://127.0.0.1:{sink.server_port}/destination"

        class Redirect(Sink):
            def do_GET(self):
                self.send_response(307)
                self.send_header("Location", target)
                self.end_headers()

        source = ThreadingHTTPServer(("127.0.0.1", 0), Redirect)
        threads = [Thread(target=server.serve_forever, daemon=True) for server in (sink, source)]
        for thread in threads:
            thread.start()
        try:
            request = Request(f"http://127.0.0.1:{source.server_port}/start",
                              headers={"Authorization": "Bearer test-credential"})
            with self.assertRaises(HTTPError) as error:
                acceptance.HTTP.open(request, timeout=5)
            self.assertEqual(error.exception.code, 307)
            self.assertEqual(received, [])
        finally:
            for server in (source, sink):
                server.shutdown()
                server.server_close()
            for thread in threads:
                thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
