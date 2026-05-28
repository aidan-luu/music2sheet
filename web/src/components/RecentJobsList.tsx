import { Link } from 'react-router-dom';
import type { RecentJob } from '../lib/recentJobs';

interface Props {
  jobs: ReadonlyArray<RecentJob>;
  onClear: () => void;
}

function formatTimestamp(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString();
}

export function RecentJobsList({ jobs, onClear }: Props): JSX.Element {
  return (
    <aside className="recent-jobs" aria-label="Recent jobs">
      <div className="recent-jobs__header">
        <h2>Recent jobs</h2>
        {jobs.length > 0 && (
          <button
            type="button"
            className="recent-jobs__clear"
            onClick={onClear}
            aria-label="Clear recent jobs history"
          >
            Clear history
          </button>
        )}
      </div>
      {jobs.length === 0 ? (
        <p className="recent-jobs__empty">No jobs yet. Submit one to get started.</p>
      ) : (
        <ul className="recent-jobs__list">
          {jobs.map((job) => (
            <li key={job.id} className="recent-jobs__item">
              <Link to={`/jobs/${job.id}`} className="recent-jobs__link">
                <span className="recent-jobs__label" title={job.inputLabel}>
                  {job.inputLabel}
                </span>
                <span className="recent-jobs__time">{formatTimestamp(job.createdAt)}</span>
              </Link>
            </li>
          ))}
        </ul>
      )}
    </aside>
  );
}
