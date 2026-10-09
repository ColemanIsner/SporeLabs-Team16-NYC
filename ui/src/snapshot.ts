// Static snapshot for the public page (tools/snapshot_story.py). Built with VITE_SNAPSHOT=1, the Story
// reads recorded JSON, recorded VSS searches and /media/spore-*.mp4 instead of the live backend.
export type Snapshot = {
  recorded: string;
  json: Record<string, unknown>;
  media: Record<string, string>;
  ask: Record<string, unknown>;
};

const found = import.meta.glob("../snapshot/snapshot.json", { eager: true, import: "default" });
export const SNAP: Snapshot | null =
  import.meta.env.VITE_SNAPSHOT === "1" ? ((Object.values(found)[0] as Snapshot | undefined) ?? null) : null;
