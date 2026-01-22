# Copyright 2026 FBK

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
PROTO_DIR = Path(ROOT / "simulstream/server/speech_processors/rpc")


class build_py(_build_py):
    def run(self):
        cmd = [
            sys.executable,
            "-m",
            "grpc_tools.protoc",
            f"-I{PROTO_DIR}",
            f"--python_out={PROTO_DIR}",
            f"--grpc_python_out={PROTO_DIR}",
            str(PROTO_DIR / "speech_processor.proto"),
        ]

        subprocess.check_call(cmd)

        for file in PROTO_DIR.glob("*_pb2*.py"):
            text = file.read_text()
            text = text.replace(
                "import speech_processor_pb2 as",
                "from . import speech_processor_pb2 as",
            )
            file.write_text(text)
        super().run()


setup(cmdclass={"build_py": build_py})
