import { useEffect, useMemo, useState } from 'react'
import { marked } from 'marked'
import './DocsApp.css'

import indexMd from '../docs/README.md?raw'
import planMd from '../docs/OFFICIAL_DOCUMENTATION_PLAN.md?raw'
import grammarMd from '../docs/02-grammar/syntax.md?raw'
import transpilerMd from '../docs/04-toolchain/transpiler.md?raw'
import compilerMd from '../docs/04-toolchain/compiler.md?raw'
import examplesMd from '../docs/05-examples/canonical-examples.md?raw'
import llmMd from '../docs/07-llm/llm-contract.md?raw'

type DocItem = {
  id: string
  title: string
  section: string
  content: string
}

const DOCS: DocItem[] = [
  { id: 'index', title: 'Docs Index', section: 'Start', content: indexMd },
  { id: 'plan', title: 'Official Plan', section: 'Start', content: planMd },
  { id: 'grammar', title: 'Grammar Syntax', section: 'Grammar', content: grammarMd },
  { id: 'transpiler', title: 'Transpiler', section: 'Toolchain', content: transpilerMd },
  { id: 'compiler', title: 'Compiler', section: 'Toolchain', content: compilerMd },
  { id: 'llm-contract', title: 'LLM Contract', section: 'LLM', content: llmMd },
  { id: 'canonical-examples', title: 'Canonical Examples', section: 'Examples', content: examplesMd },
]

const GROUPS = ['Start', 'Grammar', 'Toolchain', 'LLM', 'Examples']

function readHashSelection(): string {
  const raw = window.location.hash.replace(/^#/, '').trim()
  if (!raw) return 'index'
  return DOCS.some((d) => d.id === raw) ? raw : 'index'
}

function DocsApp() {
  const [selected, setSelected] = useState<string>(readHashSelection)

  useEffect(() => {
    const onHashChange = () => setSelected(readHashSelection())
    window.addEventListener('hashchange', onHashChange)
    return () => window.removeEventListener('hashchange', onHashChange)
  }, [])

  const selectedDoc = useMemo(
    () => DOCS.find((doc) => doc.id === selected) ?? DOCS[0],
    [selected],
  )

  const html = useMemo(() => {
    return marked.parse(selectedDoc.content, {
      gfm: true,
      breaks: false,
    }) as string
  }, [selectedDoc])

  function openDoc(id: string) {
    window.location.hash = id
    setSelected(id)
  }

  return (
    <div className="docs-page">
      <header className="docs-topbar">
        <div>
          <p className="docs-kicker">Kern Documentation</p>
          <h1>Official Docs (HTML View)</h1>
        </div>
        <div className="docs-actions">
          <a className="docs-btn" href="/">
            Open Studio
          </a>
          <a className="docs-btn ghost" href="https://github.com/OscarCode9/kern/tree/main/web/docs" target="_blank" rel="noreferrer">
            View Markdown
          </a>
        </div>
      </header>

      <main className="docs-layout">
        <aside className="docs-sidebar">
          {GROUPS.map((group) => {
            const items = DOCS.filter((doc) => doc.section === group)
            if (!items.length) return null
            return (
              <section key={group} className="docs-group">
                <h2>{group}</h2>
                {items.map((doc) => (
                  <button
                    key={doc.id}
                    type="button"
                    className={`docs-link ${selected === doc.id ? 'active' : ''}`}
                    onClick={() => openDoc(doc.id)}
                  >
                    {doc.title}
                  </button>
                ))}
              </section>
            )
          })}
        </aside>

        <article className="docs-article">
          <div className="docs-markdown" dangerouslySetInnerHTML={{ __html: html }} />
        </article>
      </main>
    </div>
  )
}

export default DocsApp
