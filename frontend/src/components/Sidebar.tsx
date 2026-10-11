import { NavLink, useLocation } from 'react-router-dom';
import { useEffect, useState } from 'react';
import { getStatus, type AppStatus } from '../api';

const navItems = [
  { to: '/', icon: '📊', label: 'Overview' },
  { to: '/analyze', icon: '🔬', label: 'Analyze Images' },
  { to: '/dataset', icon: '📁', label: 'Dataset' },
  { to: '/evaluation', icon: '📈', label: 'Model Evaluation' },
  { to: '/runs', icon: '⚙️', label: 'Model Runs' },
  { to: '/about', icon: 'ℹ️', label: 'About & Limitations' },
];

export default function Sidebar() {
  const location = useLocation();
  const [status, setStatus] = useState<AppStatus | null>(null);

  useEffect(() => {
    getStatus().then(setStatus).catch(() => {});
  }, []);

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <div className="sidebar-logo">
          <div className="sidebar-logo-icon">🔬</div>
          <div>
            <span className="sidebar-logo-text">Microplastic</span>
            <span className="sidebar-logo-sub">Vision</span>
          </div>
        </div>
      </div>

      <nav className="sidebar-nav">
        <span className="nav-section-label">Navigation</span>
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === '/'}
            className={({ isActive }) =>
              `nav-item ${isActive ? 'active' : ''}`
            }
          >
            <span className="nav-item-icon">{item.icon}</span>
            {item.label}
          </NavLink>
        ))}
      </nav>

      <div className="sidebar-footer">
        {status?.demo_mode && (
          <div className="demo-badge">
            <span>⚡</span> Demo Mode
          </div>
        )}
        {status && !status.demo_mode && status.model_loaded && (
          <div className="badge badge-success" style={{ fontSize: '11px' }}>
            Model: {status.active_model || 'loaded'}
          </div>
        )}
        {status && !status.demo_mode && !status.model_loaded && (
          <div className="badge badge-warning" style={{ fontSize: '11px' }}>
            No model loaded
          </div>
        )}
      </div>
    </aside>
  );
}
