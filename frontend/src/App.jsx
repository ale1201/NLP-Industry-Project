import { useCallback, useEffect, useRef, useState } from 'react'

const VERDICT_COPY = {
  clean: {
    label: 'No injection detected',
    blurb: 'Nothing in this document looks like an instruction aimed at a model.',
  },
  suspicious: {
    label: 'Suspicious',
    blurb: 'Some signals fired but the evidence is not conclusive. Review below.',
  },
  injected: {
    label: 'Prompt injection detected',
    blurb: 'This document contains content that tries to steer a language model.',
  },
}

function Meter({ value, verdict }) {
  return (
    <div className="meter" role="img" aria-label={`Risk ${(value * 100).toFixed(0)} of 100`}>
      <div className={`meter-fill ${verdict}`} style={{ width: `${Math.max(value * 100, 2)}%` }} />
    </div>
  )
}

function Signals({ signals }) {
  if (!signals.length) {
    return (
      <p className="empty">
        No structural anomalies. The page contains no hidden, invisible or
        off-page text.
      </p>
    )
  }
  return (
    <ul className="signals">
      {signals.map((s, i) => (
        <li key={`${s.id}-${i}`} className={`signal ${s.severity}`}>
          <div className="signal-head">
            <span className={`sev ${s.severity}`}>{s.severity}</span>
            <strong>{s.title}</strong>
            {s.page != null && <span className="page">page {s.page}</span>}
          </div>
          <p>{s.detail}</p>
          {s.evidence && <pre className="evidence">{s.evidence}</pre>}
        </li>
      ))}
    </ul>
  )
}

export default function App() {
  const [file, setFile] = useState(null)
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [dragging, setDragging] = useState(false)
  const [health, setHealth] = useState(null)
  const inputRef = useRef(null)

  useEffect(() => {
    fetch('/api/health')
      .then((r) => (r.ok ? r.json() : null))
      .then(setHealth)
      .catch(() => setHealth(null))
  }, [])

  const analyse = useCallback(async (f) => {
    setBusy(true)
    setError('')
    setResult(null)
    try {
      const body = new FormData()
      body.append('file', f)
      const res = await fetch('/api/analyze', { method: 'POST', body })
      const payload = await res.json().catch(() => ({}))
      if (!res.ok) throw new Error(payload.detail || `Request failed (${res.status})`)
      setResult(payload)
    } catch (e) {
      setError(e.message || 'Something went wrong.')
    } finally {
      setBusy(false)
    }
  }, [])

  const pick = useCallback(
    (f) => {
      if (!f) return
      setFile(f)
      analyse(f)
    },
    [analyse],
  )

  const onDrop = useCallback(
    (e) => {
      e.preventDefault()
      setDragging(false)
      pick(e.dataTransfer.files?.[0])
    },
    [pick],
  )

  const verdict = result ? VERDICT_COPY[result.verdict] : null

  return (
    <div className="page">
      <header>
        <h1>PDF Prompt-Injection Detector</h1>
        <p className="sub">
          Upload a PDF. Its text is scored by the NLP-Industry-Project TF-IDF
          classifier and the page is checked for text hidden from human readers
          but visible to a model.
        </p>
      </header>

      <div
        className={`drop ${dragging ? 'over' : ''}`}
        onDragOver={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => e.key === 'Enter' && inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept="application/pdf,.pdf"
          hidden
          onChange={(e) => pick(e.target.files?.[0])}
        />
        <p className="drop-title">{file ? file.name : 'Drop a PDF here'}</p>
        <p className="drop-hint">or click to browse</p>
      </div>

      {busy && <p className="status">Analysing…</p>}
      {error && <p className="error">{error}</p>}

      {result && (
        <section className="result">
          <div className={`verdict ${result.verdict}`}>
            <div className="verdict-top">
              <h2>{verdict.label}</h2>
              <span className="risk">{(result.risk * 100).toFixed(0)}<small>/100</small></span>
            </div>
            <Meter value={result.risk} verdict={result.verdict} />
            <p>{verdict.blurb}</p>
          </div>

          <dl className="facts">
            <div><dt>Pages</dt><dd>{result.pages}</dd></div>
            <div><dt>Characters</dt><dd>{result.char_count.toLocaleString()}</dd></div>
            <div><dt>Classifier score</dt><dd>{result.model_score.toFixed(3)}</dd></div>
            <div><dt>Time</dt><dd>{result.elapsed_ms} ms</dd></div>
          </dl>

          <h3>Structural signals</h3>
          <Signals signals={result.signals} />

          {result.top_chunks?.length > 0 && (
            <>
              <h3>Highest-scoring passages</h3>
              <ul className="chunks">
                {result.top_chunks.map((c) => (
                  <li key={c.index}>
                    <span className="score">{c.score.toFixed(3)}</span>
                    <p>{c.preview}</p>
                  </li>
                ))}
              </ul>
            </>
          )}
        </section>
      )}

      <footer>
        {health ? (
          <>
            Model: <code>{health.model_name}</code>
            {health.semantic_model_available === false && (
              <span className="warn"> — model not loaded; check the backend</span>
            )}
          </>
        ) : (
          <span className="warn">Backend unreachable — is uvicorn running on :8000?</span>
        )}
      </footer>
    </div>
  )
}
