import { useState, useEffect } from "react";

/** Floating action button that appears on scroll to return to the top of the page. */
export default function ScrollToTop() {
  const [visible, setVisible] = useState(false);

  /** Track scroll position to toggle button visibility. */
  useEffect(() => {
    const onScroll = () => setVisible(window.scrollY > 120);
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <button
      onClick={() => window.scrollTo({ top: 0, behavior: "smooth" })}
      className={`fixed bottom-6 right-6 z-50 w-10 h-10 flex items-center justify-center rounded-full border transition-all duration-200 cursor-pointer ${
        visible
          ? "opacity-100 pointer-events-auto border-accent bg-accent text-white shadow-lg hover:bg-accent-2 hover:border-accent-2"
          : "opacity-0 pointer-events-none border-rule bg-paper text-neutral"
      }`}
      aria-label="Scroll to top"
    >
      <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M4 10l4-4 4 4" />
      </svg>
    </button>
  );
}
