import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Search, ArrowRight, Globe, Zap, Shield } from 'lucide-react';

const LandingPage: React.FC = () => {
  const [url, setUrl] = useState('');
  const [isError, setIsError] = useState(false);
  const navigate = useNavigate();

  // const handleSubmit = (e: React.FormEvent) => {
  //   e.preventDefault();
    
  //   // Simple URL validation
  //   if (!url || !url.match(/^(https?:\/\/)?([\w-]+\.)+[\w-]+(\/[\w- .\/?%&=]*)?$/)) {
  //     setIsError(true);
  //     return;
  //   }
    
  //   setIsError(false);
  //   navigate('/processing', { state: { url } });
  // };

const handleSubmit = async (e: React.FormEvent) => {
  e.preventDefault();

  if (!url || !url.match(/^(https?:\/\/)?([\w-]+\.)+[\w-]+(\/[\w- .\/?%&=]*)?$/)) {
    setIsError(true);
    return;
  }

  setIsError(false);

  try {
    // Send request to backend
    const response = await fetch("http://localhost:5000/scrape", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });

    if (response.ok) {
      const data = await response.json(); // Get scraped data
      console.log("Scraping result:", data);

      // Navigate to /processing and pass scraped data
      navigate("/processing", { state: { url, result: data } });
    } else {
      console.error("Failed to fetch from backend");
    }
  } catch (error) {
    console.error("Error:", error);
  }
};


  const features = [
    {
      icon: <Globe className="h-8 w-8 text-indigo-500" />,
      title: 'Website Indexing',
      description: 'Automatically crawls and indexes your entire website content.'
    },
    {
      icon: <Zap className="h-8 w-8 text-indigo-500" />,
      title: 'Lightning Fast',
      description: 'Delivers search results in milliseconds with advanced caching.'
    },
    {
      icon: <Shield className="h-8 w-8 text-indigo-500" />,
      title: 'Secure & Private',
      description: 'Your data remains private and secure with end-to-end encryption.'
    }
  ];

  return (
    <div className="max-w-6xl mx-auto">
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="flex flex-col items-center justify-center min-h-[80vh] text-center px-4"
      >
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ delay: 0.2, duration: 0.5 }}
          className="mb-8"
        >
          <Search className="h-16 w-16 text-indigo-600 mx-auto" />
        </motion.div>
        
        <motion.h1 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3, duration: 0.5 }}
          className="text-4xl md:text-5xl lg:text-6xl font-bold mb-6 bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent"
        >
          Generate a Powerful Search Engine
        </motion.h1>
        
        <motion.p 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4, duration: 0.5 }}
          className="text-xl text-gray-600 mb-12 max-w-2xl"
        >
          Transform your website with an intelligent search solution. Enter your URL below to get started.
        </motion.p>
        
        <motion.form 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5, duration: 0.5 }}
          onSubmit={handleSubmit}
          className="w-full max-w-xl"
        >
          <div className="relative">
            <input
              type="text"
              value={url}
              onChange={(e) => {
                setUrl(e.target.value);
                if (isError) setIsError(false);
              }}
              placeholder="Enter your website URL (e.g., example.com)"
              className={`w-full px-6 py-4 pr-36 rounded-full border-2 ${
                isError ? 'border-red-500' : 'border-gray-200 focus:border-indigo-500'
              } outline-none text-lg transition-all duration-300 shadow-sm focus:shadow-md`}
            />
            <button
              type="submit"
              className="absolute right-2 top-2 bottom-2 bg-gradient-to-r from-indigo-600 to-purple-600 text-white px-6 rounded-full flex items-center justify-center space-x-2 hover:from-indigo-700 hover:to-purple-700 transition-all duration-300"
            >
              <span className="hidden md:inline">Generate</span>
              <ArrowRight className="h-5 w-5" />
            </button>
          </div>
          {isError && (
            <motion.p 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="text-red-500 mt-2 text-sm"
            >
              Please enter a valid website URL
            </motion.p>
          )}
        </motion.form>
        
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.7, duration: 0.5 }}
          className="mt-24 grid grid-cols-1 md:grid-cols-3 gap-8 w-full"
        >
          {features.map((feature, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.8 + index * 0.1, duration: 0.5 }}
              className="bg-white p-6 rounded-xl shadow-sm hover:shadow-md transition-shadow duration-300"
            >
              <div className="bg-indigo-50 p-3 rounded-full w-fit mx-auto mb-4">
                {feature.icon}
              </div>
              <h3 className="text-xl font-semibold mb-2">{feature.title}</h3>
              <p className="text-gray-600">{feature.description}</p>
            </motion.div>
          ))}
        </motion.div>
      </motion.div>
    </div>
  );
};

export default LandingPage;