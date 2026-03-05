import { useEffect, useMemo, useState } from 'react'
import Editor, { type Monaco } from '@monaco-editor/react'
import './App.css'

type Direction = 'python-to-kern' | 'kern-to-python'
type ApiHealth = 'checking' | 'online' | 'offline'

type DataFileInfo = {
  path: string
  size_bytes: number
}

type DataFilesResponse = {
  root: string
  files: DataFileInfo[]
}

type FileContentResponse = {
  path: string
  code: string
}

type TreeNode = {
  name: string
  path: string
  kind: 'dir' | 'file'
  children: TreeNode[]
}

const STARTER_PYTHON = `def gcd(a: int, b: int) -> int:
    while b != 0:
        a, b = b, a % b
    return a
`

const STARTER_KERN = `fn gcd(a:int,b:int)->int{
while b!=0{a,b=b,a%b}
ret a
}`

const MONACO_THEME = 'kern-dark'

function configureMonaco(monaco: Monaco) {
  monaco.editor.defineTheme(MONACO_THEME, {
    base: 'vs-dark',
    inherit: true,
    rules: [
      { token: 'comment', foreground: '7d8590', fontStyle: 'italic' },
      { token: 'keyword', foreground: '56b6ff' },
      { token: 'type', foreground: '4ec9b0' },
      { token: 'number', foreground: 'f4b36a' },
      { token: 'string', foreground: '98c379' },
      { token: 'string.escape', foreground: 'e0af68' },
      { token: 'operator', foreground: 'c678dd' },
      { token: 'entity.name.function', foreground: '61afef' },
      { token: 'entity.name.class', foreground: 'e5c07b' },
      { token: 'constant', foreground: 'd19a66' },
    ],
    colors: {
      'editor.background': '#1e1e1e',
      'editorLineNumber.foreground': '#6b7484',
      'editorLineNumber.activeForeground': '#a9b4c7',
      'editorCursor.foreground': '#d8dee9',
      'editor.selectionBackground': '#2a3b57',
      'editor.inactiveSelectionBackground': '#2a3240',
    },
  })

  if (monaco.languages.getLanguages().some((lang: { id: string }) => lang.id === 'kern')) {
    return
  }

  monaco.languages.register({ id: 'kern' })
  monaco.languages.setLanguageConfiguration('kern', {
    comments: { lineComment: '#' },
    brackets: [
      ['{', '}'],
      ['[', ']'],
      ['(', ')'],
    ],
    autoClosingPairs: [
      { open: '{', close: '}' },
      { open: '[', close: ']' },
      { open: '(', close: ')' },
      { open: '"', close: '"' },
      { open: "'", close: "'" },
    ],
    surroundingPairs: [
      { open: '{', close: '}' },
      { open: '[', close: ']' },
      { open: '(', close: ')' },
      { open: '"', close: '"' },
      { open: "'", close: "'" },
    ],
  })

  monaco.languages.setMonarchTokensProvider('kern', {
    defaultToken: '',
    keywords: [
      'fn',
      'ret',
      'cls',
      'imp',
      'from',
      'if',
      'elif',
      'else',
      'for',
      'in',
      'while',
      'try',
      'exc',
      'fin',
      'with',
      'as',
      'pass',
      'break',
      'continue',
      'global',
      'nonlocal',
      'assert',
      'del',
      'raise',
      'await',
      'yld',
      'not',
      'and',
      'or',
      'is',
    ],
    typeKeywords: ['int', 'float', 'str', 'bool', 'list', 'dict', 'set', 'tuple'],
    constants: ['True', 'False', 'None'],
    operators: [
      '==',
      '!=',
      '<=',
      '>=',
      '<',
      '>',
      '=',
      '+=',
      '-=',
      '*=',
      '/=',
      '//=',
      '%=',
      '**=',
      '&=',
      '|=',
      '^=',
      '<<=',
      '>>=',
      ':=',
      '+',
      '-',
      '*',
      '/',
      '//',
      '%',
      '**',
      '&',
      '|',
      '^',
      '~',
      '&&',
      '||',
      '->',
      '->>',
    ],
    symbols: /[=><!~?:&|+*/^%-]+/,
    tokenizer: {
      root: [
        [/#.*$/, 'comment'],
        [/(fn)(\s+)([A-Za-z_]\w*)/, ['keyword', 'white', 'entity.name.function']],
        [/(cls)(\s+)([A-Za-z_]\w*)/, ['keyword', 'white', 'entity.name.class']],
        [/[A-Za-z_]\w*/, {
          cases: {
            '@keywords': 'keyword',
            '@typeKeywords': 'type',
            '@constants': 'constant',
            '@default': 'identifier',
          },
        }],
        [/\d+(\.\d+)?([eE][-+]?\d+)?/, 'number'],
        [/"([^"\\]|\\.)*$/, 'string.invalid'],
        [/'([^'\\]|\\.)*$/, 'string.invalid'],
        [/"/, 'string', '@string_double'],
        [/'/, 'string', '@string_single'],
        [/[{}()[\]]/, '@brackets'],
        [/[,:;.]/, 'delimiter'],
        [/@symbols/, { cases: { '@operators': 'operator', '@default': 'delimiter' } }],
      ],
      string_double: [
        [/[^\\"]+/, 'string'],
        [/\\./, 'string.escape'],
        [/"/, 'string', '@pop'],
      ],
      string_single: [
        [/[^\\']+/, 'string'],
        [/\\./, 'string.escape'],
        [/'/, 'string', '@pop'],
      ],
    },
  } as Parameters<typeof monaco.languages.setMonarchTokensProvider>[1])
}

function buildTree(paths: string[]): TreeNode[] {
  const root: TreeNode = { name: '', path: '', kind: 'dir', children: [] }

  for (const fullPath of paths) {
    const parts = fullPath.split('/').filter(Boolean)
    let current = root
    let currentPath = ''

    for (let i = 0; i < parts.length; i += 1) {
      const part = parts[i]
      const isFile = i === parts.length - 1
      currentPath = currentPath ? `${currentPath}/${part}` : part

      let next = current.children.find((node) => node.name === part)
      if (!next) {
        next = {
          name: part,
          path: currentPath,
          kind: isFile ? 'file' : 'dir',
          children: [],
        }
        current.children.push(next)
      }
      current = next
    }
  }

  const sortNodes = (nodes: TreeNode[]): TreeNode[] => {
    return [...nodes]
      .map((node) =>
        node.kind === 'dir' ? { ...node, children: sortNodes(node.children) } : node,
      )
      .sort((a, b) => {
        if (a.kind !== b.kind) {
          return a.kind === 'dir' ? -1 : 1
        }
        return a.name.localeCompare(b.name)
      })
  }

  return sortNodes(root.children)
}

function fileBadge(path: string): string {
  const lower = path.toLowerCase()
  if (lower.endsWith('.py')) return 'PY'
  if (lower.endsWith('.kern')) return 'KR'
  if (lower.endsWith('.jsonl')) return 'JL'
  if (lower.endsWith('.json')) return 'JS'
  return 'TXT'
}

function App() {
  const [pythonCode, setPythonCode] = useState(STARTER_PYTHON)
  const [kernCode, setKernCode] = useState(STARTER_KERN)
  const [busy, setBusy] = useState<Direction | null>(null)
  const [error, setError] = useState('')
  const [status, setStatus] = useState('')
  const [apiHealth, setApiHealth] = useState<ApiHealth>('checking')

  const [allFiles, setAllFiles] = useState<DataFileInfo[]>([])
  const [treeNodes, setTreeNodes] = useState<TreeNode[]>([])
  const [treeLoading, setTreeLoading] = useState(false)
  const [treeError, setTreeError] = useState('')
  const [expandedDirs, setExpandedDirs] = useState<Record<string, boolean>>({})
  const [selectedPath, setSelectedPath] = useState('')
  const [openingPath, setOpeningPath] = useState('')

  useEffect(() => {
    let active = true

    const check = async () => {
      try {
        const res = await fetch('/api/health', { cache: 'no-store' })
        if (active) {
          setApiHealth(res.ok ? 'online' : 'offline')
        }
      } catch {
        if (active) {
          setApiHealth('offline')
        }
      }
    }

    void check()
    const id = window.setInterval(() => {
      void check()
    }, 3000)

    return () => {
      active = false
      window.clearInterval(id)
    }
  }, [])

  const healthLabel = useMemo(() => {
    if (apiHealth === 'checking') return 'API: checking'
    if (apiHealth === 'online') return 'API: online'
    return 'API: offline'
  }, [apiHealth])

  async function refreshFileTree() {
    setTreeError('')
    setTreeLoading(true)
    try {
      const res = await fetch('/api/files/list', { cache: 'no-store' })
      const payload = (await res.json().catch(() => null)) as
        | DataFilesResponse
        | { detail?: string }
        | null

      if (!res.ok || !payload || !('files' in payload)) {
        if (res.status >= 500) {
          throw new Error(
            'API unavailable or internal error. Start with: cd web && npm run dev',
          )
        }
        const detail = payload && 'detail' in payload ? payload.detail : `Request failed (${res.status})`
        throw new Error(detail || 'Could not load file tree')
      }

      const files = payload.files
      setAllFiles(files)
      setTreeNodes(buildTree(files.map((item) => item.path)))

      setExpandedDirs((prev) => {
        const next = { ...prev }
        for (const file of files) {
          const parts = file.path.split('/')
          let current = ''
          for (let i = 0; i < parts.length - 1; i += 1) {
            current = current ? `${current}/${parts[i]}` : parts[i]
            if (next[current] === undefined) {
              next[current] = i < 2
            }
          }
        }
        return next
      })

      setApiHealth('online')
      setStatus(`Loaded ${files.length} files from data/`)
    } catch (err) {
      setApiHealth('offline')
      setTreeError(err instanceof Error ? err.message : 'Could not load file tree')
    } finally {
      setTreeLoading(false)
    }
  }

  useEffect(() => {
    void refreshFileTree()
  }, [])

  async function openDataFile(path: string) {
    setError('')
    setOpeningPath(path)
    try {
      const res = await fetch(`/api/files/content?path=${encodeURIComponent(path)}`, {
        cache: 'no-store',
      })
      const payload = (await res.json().catch(() => null)) as
        | FileContentResponse
        | { detail?: string }
        | null

      if (!res.ok || !payload || !('code' in payload)) {
        if (res.status >= 500) {
          throw new Error(
            'API unavailable or internal error. Start with: cd web && npm run dev',
          )
        }
        const detail = payload && 'detail' in payload ? payload.detail : `Request failed (${res.status})`
        throw new Error(detail || 'Could not load file')
      }

      const content = payload.code
      const lower = path.toLowerCase()

      if (lower.endsWith('.kern')) {
        setKernCode(content)
        setStatus(`Loaded ${path} into Kern editor`)
      } else {
        setPythonCode(content)
        setStatus(`Loaded ${path} into Python editor`)
      }

      setSelectedPath(path)
      setApiHealth('online')
    } catch (err) {
      setApiHealth('offline')
      setError(err instanceof Error ? err.message : 'Could not open file')
    } finally {
      setOpeningPath('')
    }
  }

  function toggleDir(path: string) {
    setExpandedDirs((prev) => ({ ...prev, [path]: !prev[path] }))
  }

  async function convert(direction: Direction) {
    setError('')
    setStatus('')

    const source = direction === 'python-to-kern' ? pythonCode : kernCode
    if (!source.trim()) {
      setError('Input is empty. Paste code first.')
      return
    }

    setBusy(direction)
    const t0 = performance.now()

    try {
      const endpoint =
        direction === 'python-to-kern'
          ? '/api/convert/python-to-kern'
          : '/api/convert/kern-to-python'

      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code: source }),
      })

      const payload = (await res.json().catch(() => null)) as
        | { code?: string; detail?: string }
        | null

      if (!res.ok) {
        const detail = payload?.detail
        if (detail) {
          throw new Error(detail)
        }
        if (res.status >= 500) {
          throw new Error(
            'API unavailable or internal error. Start backend with: python3 -m uvicorn backend.main:app --reload --port 8000',
          )
        }
        throw new Error(`Request failed with ${res.status}`)
      }

      const out = payload?.code ?? ''
      if (direction === 'python-to-kern') {
        setKernCode(out)
      } else {
        setPythonCode(out)
      }
      setApiHealth('online')

      const elapsed = performance.now() - t0
      setStatus(
        direction === 'python-to-kern'
          ? `Converted Python -> Kern in ${elapsed.toFixed(0)} ms`
          : `Converted Kern -> Python in ${elapsed.toFixed(0)} ms`,
      )
    } catch (err) {
      setApiHealth('offline')
      setError(err instanceof Error ? err.message : 'Unexpected conversion error')
    } finally {
      setBusy(null)
    }
  }

  function renderTree(nodes: TreeNode[], depth = 0): React.ReactNode {
    return nodes.map((node) => {
      const pad = 10 + depth * 14
      if (node.kind === 'dir') {
        const open = expandedDirs[node.path] ?? depth < 2
        return (
          <div key={node.path}>
            <button
              type="button"
              className="tree-item dir"
              style={{ paddingLeft: `${pad}px` }}
              onClick={() => toggleDir(node.path)}
            >
              <span className="caret">{open ? 'v' : '>'}</span>
              <span>{node.name}</span>
            </button>
            {open ? renderTree(node.children, depth + 1) : null}
          </div>
        )
      }

      const active = selectedPath === node.path
      return (
        <button
          key={node.path}
          type="button"
          className={`tree-item file ${active ? 'active' : ''}`}
          style={{ paddingLeft: `${pad + 14}px` }}
          onClick={() => void openDataFile(node.path)}
          disabled={openingPath.length > 0 && openingPath !== node.path}
        >
          <span className="file-badge">{fileBadge(node.path)}</span>
          <span className="tree-name">{node.name}</span>
        </button>
      )
    })
  }

  return (
    <div className="page">
      <header className="hero">
        <p className="kicker">Kern Studio</p>
        <h1>Python to/from Kern IDE</h1>
        <p className="subtitle">
          VS Code-like explorer for <code>data/</code> and dark themed code editing with Monaco.
        </p>
        <p className={`health ${apiHealth}`}>{healthLabel}</p>
      </header>

      <main className="ide-shell">
        <aside className="sidebar panel">
          <div className="panel-head">
            <h2>Explorer</h2>
            <button className="btn ghost" onClick={() => void refreshFileTree()} disabled={treeLoading}>
              {treeLoading ? 'Loading...' : 'Refresh'}
            </button>
          </div>
          <p className="side-meta">/Users/oscarcode/kern/data ({allFiles.length} files)</p>

          <div className="tree-wrap">
            {treeError ? <p className="error">{treeError}</p> : null}
            {!treeError && treeNodes.length === 0 && !treeLoading ? (
              <p className="side-empty">No files found in data/.</p>
            ) : null}
            {renderTree(treeNodes)}
          </div>
        </aside>

        <section className="panel editor-panel">
          <div className="panel-head">
            <h2>Python</h2>
            <button
              className="btn primary"
              onClick={() => void convert('python-to-kern')}
              disabled={busy !== null}
            >
              {busy === 'python-to-kern' ? 'Converting...' : 'Python -> Kern'}
            </button>
          </div>
          <div className="editor-host">
            <Editor
              height="52vh"
              language="python"
              theme={MONACO_THEME}
              beforeMount={configureMonaco}
              value={pythonCode}
              onChange={(value) => setPythonCode(value ?? '')}
              options={{
                minimap: { enabled: false },
                fontSize: 14,
                wordWrap: 'on',
                automaticLayout: true,
                scrollBeyondLastLine: false,
                fontFamily: 'IBM Plex Mono, Menlo, monospace',
              }}
            />
          </div>
        </section>

        <section className="panel editor-panel">
          <div className="panel-head">
            <h2>Kern</h2>
            <button
              className="btn primary"
              onClick={() => void convert('kern-to-python')}
              disabled={busy !== null}
            >
              {busy === 'kern-to-python' ? 'Converting...' : 'Kern -> Python'}
            </button>
          </div>
          <div className="editor-host">
            <Editor
              height="52vh"
              language="kern"
              theme={MONACO_THEME}
              beforeMount={configureMonaco}
              value={kernCode}
              onChange={(value) => setKernCode(value ?? '')}
              options={{
                minimap: { enabled: false },
                fontSize: 14,
                wordWrap: 'on',
                automaticLayout: true,
                scrollBeyondLastLine: false,
                fontFamily: 'IBM Plex Mono, Menlo, monospace',
              }}
            />
          </div>
        </section>
      </main>

      <footer className="statusbar">
        {error ? <p className="error">{error}</p> : <p className="ok">{status || 'Ready'}</p>}
      </footer>
    </div>
  )
}

export default App
