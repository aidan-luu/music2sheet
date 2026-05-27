import { Route, BrowserRouter, Routes, Link } from 'react-router-dom';
import { UploadPage } from './pages/UploadPage';
import { JobStatusPage } from './pages/JobStatusPage';
import './app.css';

export function App(): JSX.Element {
  return (
    <BrowserRouter>
      <div className="app">
        <header className="app__header">
          <Link to="/" className="app__brand">
            music2sheet
          </Link>
        </header>
        <main className="app__main">
          <Routes>
            <Route path="/" element={<UploadPage />} />
            <Route path="/jobs/:jobId" element={<JobStatusPage />} />
            <Route path="*" element={<p>Not found.</p>} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}
