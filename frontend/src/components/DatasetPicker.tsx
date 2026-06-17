import { useEffect, useState } from "react";
import { getDatasets } from "../api";

// Wiki dataset (version) selector. Loads the registered+built datasets from /datasets and
// reports the chosen id upward; App passes it to /ask/stream as ?dataset=<id>. If only one
// dataset is built there's nothing to choose, so the control hides itself.
export default function DatasetPicker({
  value,
  onChange,
  disabled,
}: {
  value: string;
  onChange: (id: string) => void;
  disabled?: boolean;
}) {
  const [ids, setIds] = useState<string[]>([]);

  useEffect(() => {
    let alive = true;
    getDatasets()
      .then((d) => {
        if (!alive) return;
        const built = Object.entries(d.datasets)
          .filter(([, ok]) => ok)
          .map(([id]) => id);
        setIds(built);
        // if the current selection isn't a built dataset, fall back to the server default
        if (built.length && !built.includes(value)) {
          onChange(built.includes(d.default) ? d.default : built[0]);
        }
      })
      .catch(() => {
        /* /datasets unreachable — leave the picker hidden, default dataset still used */
      });
    return () => {
      alive = false;
    };
    // run once on mount; onChange/value are intentionally not deps
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (ids.length < 2) return null; // nothing to choose between

  return (
    <label className="dataset-picker" title="위키 데이터셋(버전) 선택">
      <span>데이터셋</span>
      <select
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value)}
      >
        {ids.map((id) => (
          <option key={id} value={id}>
            {id}
          </option>
        ))}
      </select>
    </label>
  );
}
