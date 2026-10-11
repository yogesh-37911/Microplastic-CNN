import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import OverviewPage from './pages/OverviewPage';
import AnalyzePage from './pages/AnalyzePage';
import DatasetPage from './pages/DatasetPage';
import EvaluationPage from './pages/EvaluationPage';
import RunsPage from './pages/RunsPage';
import AboutPage from './pages/AboutPage';
import './index.css';

function App() {
  return (
    <BrowserRouter>
      <div className="app-layout">
        <Sidebar />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<OverviewPage />} />
            <Route path="/analyze" element={<AnalyzePage />} />
            <Route path="/dataset" element={<DatasetPage />} />
            <Route path="/evaluation" element={<EvaluationPage />} />
            <Route path="/runs" element={<RunsPage />} />
            <Route path="/about" element={<AboutPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}

export default App;
