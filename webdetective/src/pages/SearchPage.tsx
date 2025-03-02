import React, { useState, useRef, useEffect } from "react";
import { useLocation } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { Search, Send, ExternalLink, ThumbsUp, ThumbsDown, Copy, Check } from "lucide-react";

interface SearchResult {
  Link: string;      // Matches Flask API key
  Tags: string;      // Matches Flask API key
  Similarity: number; // Matches Flask API key
}

const SearchPage: React.FC = () => {
  const location = useLocation();
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [copied, setCopied] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setIsSearching(true);
    setResults([]); // Clear previous results

    try {
      const response = await fetch("http://127.0.0.1:5000/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });

      if (!response.ok) throw new Error("Failed to fetch results");

      const data = await response.json();
      setResults(data);
    } catch (error) {
      console.error("Search error:", error);
    } finally {
      setIsSearching(false);
    }
  };

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopied(id);
    setTimeout(() => setCopied(null), 2000);
  };

  useEffect(() => {
    if (inputRef.current) inputRef.current.focus();
  }, []);

  return (
    <div className="max-w-6xl mx-auto">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
        <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.5 }} className="md:col-span-1">
          <div className="bg-white rounded-xl shadow-sm p-6 sticky top-24">
            <h2 className="text-xl font-bold mb-2">Search Information</h2>
            <p className="text-gray-600 text-sm">Explore the best matches for your query.</p>
          </div>
        </motion.div>

        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, delay: 0.2 }} className="md:col-span-2">
          <div className="bg-white rounded-xl shadow-sm p-6 mb-6">
            <form onSubmit={handleSearch} className="relative">
              <input
                ref={inputRef}
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search for anything..."
                className="w-full px-4 py-3 pr-12 rounded-lg border border-gray-200 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 outline-none transition-all"
              />
              <button type="submit" className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-indigo-600 transition-colors">
                {isSearching ? (
                  <motion.div animate={{ rotate: 360 }} transition={{ duration: 1, repeat: Infinity, ease: "linear" }}>
                    <Search className="h-5 w-5" />
                  </motion.div>
                ) : (
                  <Send className="h-5 w-5" />
                )}
              </button>
            </form>
          </div>

          <AnimatePresence>
            {isSearching ? (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="space-y-4">
                {[1, 2, 3].map((i) => (
                  <div key={i} className="bg-white rounded-xl shadow-sm p-6 animate-pulse">
                    <div className="h-5 bg-gray-200 rounded w-3/4 mb-3"></div>
                    <div className="h-4 bg-gray-200 rounded w-1/2 mb-2"></div>
                    <div className="h-4 bg-gray-200 rounded w-full mb-2"></div>
                    <div className="h-4 bg-gray-200 rounded w-5/6"></div>
                  </div>
                ))}
              </motion.div>
            ) : results.length > 0 ? (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="space-y-4">
                <p className="text-sm text-gray-500 mb-2">{results.length} results for "{query}"</p>
                {results.map((result, index) => (
                  <motion.div key={index} initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: index * 0.1, duration: 0.3 }} className="bg-white rounded-xl shadow-sm p-6 hover:shadow-md transition-shadow">
                    <a href={result.Link} target="_blank" rel="noopener noreferrer" className="text-sm text-green-600 hover:underline flex items-center mb-2">
                      {result.Link}
                    </a>
                    <p className="text-gray-600 mb-4">Tags: {result.Tags}</p>
                    <span className="text-xs bg-indigo-50 text-indigo-600 px-2 py-1 rounded-full">
                      {(result.Similarity * 100).toFixed(2)}% match
                    </span>
                  </motion.div>
                ))}
              </motion.div>
            ) : query && !isSearching ? null: null}
          </AnimatePresence>
        </motion.div>
      </div>
    </div>
  );
};

export default SearchPage;
