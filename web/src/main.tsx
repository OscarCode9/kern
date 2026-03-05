import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import DocsApp from './DocsApp.tsx'

const path = window.location.pathname.replace(/\/+$/, '') || '/'
const params = new URLSearchParams(window.location.search)
const isDocsRoute = path === '/docs' || params.get('view') === 'docs'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    {isDocsRoute ? <DocsApp /> : <App />}
  </StrictMode>,
)
