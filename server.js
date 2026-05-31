import express from 'express';
import cors from 'cors';
import fetch from 'node-fetch';
globalThis.fetch = fetch;

const app = express();
const port = 3001;

// Middleware
app.use(cors()); // Allows your frontend to talk to this server
// Increase body size limit to support large prompts
app.use(express.json({ limit: '20mb' })); // Parses JSON bodies
app.use(express.urlencoded({ limit: '20mb', extended: true }));
app.use(express.static('.')); // Serve static files from current directory

// Chat Endpoint
app.post('/chat', async (req, res) => {
    const { message } = req.body;

    if (!message) {
        return res.status(400).json({ error: 'Message is required' });
    }

    try {
        // Use AbortController to enforce a long fetch timeout (e.g., 5 minutes)
        const controller = new AbortController();
        const FETCH_TIMEOUT_MS = 5 * 60 * 1000; // 5 minutes
        let timeout; // declared so catch/finally can clear it
        timeout = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);

        const response = await fetch('http://localhost:11434/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Connection': 'keep-alive'
            },
            body: JSON.stringify({
                model: 'mistral',
                messages: [{ role: 'user', content: message }],
                stream: true,
                temperature: 0.7,
                top_p: 0.9
            }),
            signal: controller.signal
        });
        clearTimeout(timeout);

        if (!response.ok) {
            return res.status(500).json({ error: 'Failed to fetch response from Ollama' });
        }

        // Set response headers for streaming
        res.setHeader('Content-Type', 'application/x-ndjson');
        res.setHeader('Transfer-Encoding', 'chunked');
        // Keep client connection alive while processing
        res.setHeader('Connection', 'keep-alive');
        // Immediately flush headers to the client
        if (res.flushHeaders) res.flushHeaders();

        // Properly stream the response. Support both WHATWG streams (getReader)
        // and Node.js readable streams (response.body is EventEmitter).
        const decoder = new TextDecoder();
        let buffer = '';

        if (response.body && typeof response.body.getReader === 'function') {
            // WHATWG ReadableStream (browser-like)
            const reader = response.body.getReader();
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';
                for (const line of lines) {
                    if (line.trim()) {
                        try {
                            const data = JSON.parse(line);
                            res.write(JSON.stringify(data) + '\n');
                        } catch (e) {
                            // ignore non-json lines
                        }
                    }
                }
            }
            if (buffer.trim()) {
                try { const data = JSON.parse(buffer); res.write(JSON.stringify(data) + '\n'); } catch(e){}
            }
            res.end();
        } else if (response.body && typeof response.body.on === 'function') {
            // Node.js Readable stream
            response.body.on('data', (chunk) => {
                try {
                    buffer += decoder.decode(chunk, { stream: true });
                } catch (e) {
                    buffer += chunk.toString();
                }
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';
                for (const line of lines) {
                    if (line.trim()) {
                        try {
                            const data = JSON.parse(line);
                            res.write(JSON.stringify(data) + '\n');
                        } catch (e) {
                            // ignore non-json lines
                        }
                    }
                }
            });

            response.body.on('end', () => {
                if (buffer.trim()) {
                    try { const data = JSON.parse(buffer); res.write(JSON.stringify(data) + '\n'); } catch(e){}
                }
                try { res.end(); } catch(e){}
            });

            response.body.on('error', (err) => {
                console.error('Stream error from Ollama proxy:', err);
                if (!res.headersSent) {
                    res.status(500).json({ error: 'Stream error from upstream' });
                } else {
                    try { res.end(); } catch (e) {}
                }
            });

            // Wait for end/error before continuing
            await new Promise((resolve, reject) => {
                response.body.on('end', resolve);
                response.body.on('error', reject);
            });
        } else {
            // Fallback: read text and send
            const txt = await response.text();
            res.send(txt);
        }
    } catch (error) {
        try { if (typeof timeout !== 'undefined') clearTimeout(timeout); } catch(e){}
        console.error('Ollama Error:', error);
        if (error && (error.name === 'AbortError' || error.type === 'aborted')) {
            return res.status(504).json({ error: 'Upstream request timed out' });
        }
        res.status(500).json({ error: 'Failed to connect to Ollama' });
    }
});

// Simple in-memory document store and vector index for RAG
const docs = new Map(); // id => { id, title, text, chunks: [{id, text, vec}] }

// Simple LRU cache for query results
function createLRU(max = 200) {
    const map = new Map();
    return {
        get(k) { const v = map.get(k); if (v !== undefined) { map.delete(k); map.set(k, v); } return v; },
        set(k, v) { if (map.has(k)) map.delete(k); map.set(k, v); while (map.size > max) { const first = map.keys().next().value; map.delete(first); } },
        has(k) { return map.has(k); }
    };
}

const ragCache = createLRU(300);

function chunkText(text, size = 800, overlap = 200) {
    const chunks = [];
    let i = 0;
    while (i < text.length) {
        const chunk = text.slice(i, i + size);
        chunks.push(chunk.trim());
        i += size - overlap;
    }
    return chunks;
}

function tokenize(text) {
    return text.toLowerCase().replace(/[^a-z0-9\s]/g, ' ').split(/\s+/).filter(Boolean);
}

function vectorize(text) {
    const toks = tokenize(text);
    const vec = new Map();
    for (const t of toks) vec.set(t, (vec.get(t) || 0) + 1);
    // normalize to unit vector
    let sumSq = 0;
    for (const v of vec.values()) sumSq += v * v;
    const norm = Math.sqrt(sumSq) || 1;
    for (const k of vec.keys()) vec.set(k, vec.get(k) / norm);
    return vec;
}

function cosine(a, b) {
    let sum = 0;
    for (const [k, v] of a) {
        if (b.has(k)) sum += v * b.get(k);
    }
    return sum; // since vectors normalized, this is cosine
}

// Add document endpoint: stores chunks and vectors
app.post('/documents', (req, res) => {
    const { title, text } = req.body;
    if (!text) return res.status(400).json({ error: 'text required' });
    const id = `doc_${Date.now()}`;
    const chunks = chunkText(text, 800, 200).map((c, idx) => ({ id: `${id}_c${idx}`, text: c, vec: vectorize(c) }));
    docs.set(id, { id, title: title || id, text, chunks });
    return res.json({ id, chunks: chunks.length });
});

app.get('/documents', (req, res) => {
    const list = [];
    for (const [id, doc] of docs.entries()) list.push({ id, title: doc.title, chunks: doc.chunks.length });
    res.json(list);
});

// RAG query: retrieve topK chunks then call model
app.post('/rag', async (req, res) => {
    const { query, docId, topK = 3 } = req.body;
    if (!query) return res.status(400).json({ error: 'query required' });
    if (!docId) return res.status(400).json({ error: 'docId required' });
    const doc = docs.get(docId);
    if (!doc) return res.status(404).json({ error: 'document not found' });

    const cacheKey = `${docId}::${query}`;
    if (ragCache.has(cacheKey)) {
        return res.json({ cached: true, result: ragCache.get(cacheKey) });
    }

    const qvec = vectorize(query);
    const scored = doc.chunks.map(c => ({ id: c.id, text: c.text, score: cosine(qvec, c.vec) }));
    scored.sort((a, b) => b.score - a.score);
    const top = scored.slice(0, topK).filter(s => s.score > 0);

    // Build context
    const context = top.map(t => `Source: ${t.id}\n${t.text}`).join('\n\n');
    const prompt = `You are an expert financial analyst. Use the following context extracted from a document to answer the question.\n\nCONTEXT:\n${context}\n\nQUESTION:\n${query}`;

    try {
        // Call model (non-stream) for RAG with retries/backoff
        const MAX_RETRIES = 2; // total attempts = MAX_RETRIES + 1
        const FETCH_TIMEOUT_MS = 2 * 60 * 1000; // 2 minutes per attempt
        let attempt = 0;
        let lastErr = null;

        while (attempt <= MAX_RETRIES) {
            attempt++;
            const controller = new AbortController();
            let timeout;
            timeout = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);

            try {
                const response = await fetch('http://localhost:11434/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ model: 'mistral', messages: [{ role: 'user', content: prompt }], stream: false }),
                    signal: controller.signal
                });
                clearTimeout(timeout);

                if (!response.ok) {
                    lastErr = new Error('model failed: ' + response.status);
                    console.error(`RAG attempt ${attempt} received non-OK status:`, response.status);
                    if (response.status >= 500 && attempt <= MAX_RETRIES) {
                        await new Promise(r => setTimeout(r, 1000 * attempt));
                        continue; // retry on server errors
                    }
                    return res.status(500).json({ error: 'model failed' });
                }

                const data = await response.json();
                // Cache and return
                ragCache.set(cacheKey, data);
                return res.json({ cached: false, result: data });
            } catch (errInner) {
                try { if (typeof timeout !== 'undefined') clearTimeout(timeout); } catch(e){}
                console.error(`RAG attempt ${attempt} error:`, errInner && errInner.message ? errInner.message : errInner);
                lastErr = errInner;

                // If aborted due to timeout, retry a few times then return 504
                if (errInner && (errInner.name === 'AbortError' || errInner.type === 'aborted')) {
                    if (attempt <= MAX_RETRIES) {
                        await new Promise(r => setTimeout(r, 1000 * attempt));
                        continue;
                    } else {
                        return res.status(504).json({ error: 'Upstream request timed out' });
                    }
                }

                // For other transient errors, retry a few times
                if (attempt <= MAX_RETRIES) {
                    await new Promise(r => setTimeout(r, 1000 * attempt));
                    continue;
                }

                return res.status(500).json({ error: 'RAG model error' });
            }
        }

        console.error('RAG failed after retries', lastErr);
        return res.status(500).json({ error: 'RAG model error' });
    } catch (err) {
        console.error('RAG model error', err);
        return res.status(500).json({ error: 'RAG model error' });
    }
});

app.listen(port, () => {
    console.log(`🚀 Server running at http://localhost:${port}`);
});
