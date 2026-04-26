import { useState, useEffect } from "react";

interface GallerySession {
  session_id: string;
  model: string;
  duration_ms: number;
  timestamp: string;
  images: (string | null)[];
  labels: string[];
  original?: string;
}

export function Gallery() {
  const [sessions, setSessions] = useState<GallerySession[]>([]);
  const [loading, setLoading] = useState(true);
  const [expanded, setExpanded] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/gallery")
      .then((r) => r.json())
      .then((data) => {
        setSessions(data.sessions || []);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  if (loading) {
    return <p className="text-gray-500 text-sm">Loading gallery...</p>;
  }

  if (sessions.length === 0) {
    return (
      <div className="rounded-xl border border-dashed border-gray-700 p-8 text-center">
        <p className="text-gray-400">No generations yet</p>
        <p className="text-xs text-gray-500 mt-1">
          Generated hairstyles will appear here
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <h3 className="text-sm font-medium text-gray-300">
        Past Generations ({sessions.length})
      </h3>
      {sessions.map((session) => {
        const isExpanded = expanded === session.session_id;
        const modelName = session.model.split("/").pop();
        const date = new Date(session.timestamp).toLocaleString();

        return (
          <div
            key={session.session_id}
            className="rounded-xl border border-gray-800 bg-gray-900 overflow-hidden"
          >
            <button
              onClick={() =>
                setExpanded(isExpanded ? null : session.session_id)
              }
              className="w-full flex items-center justify-between px-4 py-3 hover:bg-gray-800 transition text-left"
            >
              <div className="flex items-center gap-3">
                {session.original && (
                  <img
                    src={session.original}
                    alt="Original"
                    className="h-10 w-10 rounded-lg object-cover"
                  />
                )}
                <div>
                  <p className="text-sm text-white">{modelName}</p>
                  <p className="text-xs text-gray-500">
                    {date} — {(session.duration_ms / 1000).toFixed(1)}s
                  </p>
                </div>
              </div>
              <span className="text-gray-500 text-xs">
                {isExpanded ? "Hide" : "Show"}
              </span>
            </button>

            {isExpanded && (
              <div className="px-4 pb-4">
                <div className="grid grid-cols-2 gap-2">
                  {session.images.map((url, i) =>
                    url ? (
                      <div key={i} className="space-y-1">
                        <img
                          src={url}
                          alt={session.labels[i]}
                          className="w-full rounded-lg object-cover aspect-square"
                        />
                        <div className="flex items-center justify-between">
                          <p className="text-xs text-gray-400">
                            {session.labels[i]}
                          </p>
                          <a
                            href={url}
                            download={`${session.labels[i]}.png`}
                            className="text-xs text-indigo-400 hover:text-indigo-300"
                          >
                            Save
                          </a>
                        </div>
                      </div>
                    ) : (
                      <div
                        key={i}
                        className="aspect-square rounded-lg bg-gray-800 flex items-center justify-center"
                      >
                        <p className="text-xs text-gray-600">Failed</p>
                      </div>
                    )
                  )}
                </div>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
