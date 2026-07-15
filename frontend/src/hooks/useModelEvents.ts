interface GroundedSamState {
  loaded: boolean;
  status: string;
}

interface ModelStates {
  groundedSam: GroundedSamState;
}

const defaults: ModelStates = {
  groundedSam: { loaded: false, status: "unloaded" },
};

let cached: ModelStates = { ...defaults };
let subscribers: Array<(s: ModelStates) => void> = [];
let es: EventSource | null = null;
let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

function connect() {
  if (subscribers.length === 0) return; // no active subscribers, don't reconnect
  if (es) return;
  es = new EventSource(`${API_BASE}/model/events`);
  es.onmessage = (e) => {
    try {
      const data = JSON.parse(e.data);
      cached = { ...defaults, ...data };
      subscribers.forEach((fn) => fn(cached));
    } catch {
      /* ignore */
    }
  };
  es.onerror = () => {
    es?.close();
    es = null;
    if (reconnectTimer) clearTimeout(reconnectTimer);
    reconnectTimer = setTimeout(connect, 5000);
  };
}

function subscribe(fn: (s: ModelStates) => void) {
  subscribers.push(fn);
  if (subscribers.length === 1) connect();
  fn(cached);
  return () => {
    subscribers = subscribers.filter((s) => s !== fn);
    if (subscribers.length === 0) {
      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
      es?.close();
      es = null;
    }
  };
}

/** Immediately mark a model as loading — SSE will correct it on next poll.
    Skips if the model is already loaded to avoid flickering. */
export function optimisticModelLoading() {
  if (cached.groundedSam.status === "loaded") return;
  cached = { ...cached, groundedSam: { ...cached.groundedSam, status: "loading" } };
  subscribers.forEach((fn) => fn(cached));
}

/** Immediately mark a model as unloaded — SSE will correct it on next poll. */
export function optimisticModelUnloaded() {
  cached = { ...cached, groundedSam: { loaded: false, status: "unloaded" } };
  subscribers.forEach((fn) => fn(cached));
}

export function useModelEvents(): ModelStates {
  const [state, setState] = useState<ModelStates>(cached);
  useEffect(() => subscribe(setState), []);
  return state;
}
