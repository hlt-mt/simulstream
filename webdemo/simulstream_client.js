/*
 * Copyright 2025 FBK
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


function float32ToInt16(float32) {
    const int16 = new Int16Array(float32.length);
    for (let i = 0; i < float32.length; i++) {
    let s = float32[i] * 32768;
        int16[i] = Math.max(-32768, Math.min(32767, s));
    }
    return int16;
}


export class SimulStreamClient {
    constructor(url) {
        this.url = url;
        this.socket = null;

        // callbacks available for the user to define custom logic
        this.onopen = null;
        this.onerror = null;
        this.onservermessage = null;

        // audio recording variables
        this.audioContext = null;
        this.streamSource = null;
        this.processorNode = null;
        this.stream = null;
    }

    async start(tgt_lang, src_lang) {
        // setup audio capture
        this.stream = await navigator.mediaDevices.getUserMedia({audio: true});

        this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        this.streamSource = this.audioContext.createMediaStreamSource(this.stream);

        // setup websocket connection
        console.assert(this.socket == null);
        this.socket = new WebSocket(this.url);

        if (this.onerror) this.socket.onerror = this.onerror;

        this.socket.onopen = () => {
            if (this.onopen) this.onopen();
            this.socket.send(JSON.stringify({
                target_lang: tgt_lang,
                source_lang: src_lang,
                sample_rate: this.audioContext.sampleRate
            }));
        };

        this.socket.onmessage = (event) => {
            const server_message = JSON.parse(event.data);
            if (this.onservermessage) this.onservermessage(server_message);
        };

        await this.audioContext.audioWorklet.addModule('processor.js');

        this.processorNode = new AudioWorkletNode(this.audioContext, 'pcm-processor');
        this.streamSource.connect(this.processorNode).connect(this.audioContext.destination);

        this.processorNode.port.onmessage = (event) => {
            if (event.data.length > 0 && this.socket.readyState === WebSocket.OPEN) {
                const int16 = float32ToInt16(event.data);
                this.socket.send(int16.buffer);
            }
        };
    }

    stop() {
        if (this.processorNode) this.processorNode.disconnect();
        if (this.audioContext) this.audioContext.close();
        if (this.stream) this.stream.getTracks().forEach(track => track.stop());
    }

    closeSocket() {
        if (this.socket) {
            this.socket.close();
            this.socket = null;
        }
    }

    socketOpen() {
        return this.socket !== null;
    }
}
