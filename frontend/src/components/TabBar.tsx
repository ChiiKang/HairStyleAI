interface Props {
  activeTab: string;
  onChange: (tab: string) => void;
}

const TABS = [
  { id: "color", label: "Color Change" },
  { id: "hairstyle", label: "Hairstyle Generator" },
];

export function TabBar({ activeTab, onChange }: Props) {
  return (
    <div className="flex gap-1 rounded-lg bg-gray-900 p-1">
      {TABS.map((tab) => (
        <button
          key={tab.id}
          onClick={() => onChange(tab.id)}
          className={`rounded-md px-4 py-2 text-sm font-medium transition ${
            activeTab === tab.id
              ? "bg-indigo-600 text-white"
              : "text-gray-400 hover:text-white hover:bg-gray-800"
          }`}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}
