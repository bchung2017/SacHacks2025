import React, { useEffect, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Search, Server, Database, Code, CheckCircle } from 'lucide-react';

const ProcessingPage: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const [progress, setProgress] = useState(0);
  const [currentStage, setCurrentStage] = useState(0);
  const url = location.state?.url || 'example.com';

  const stages = [
    { name: 'Crawling website', icon: <Search className="h-6 w-6" /> },
    { name: 'Analyzing content', icon: <Code className="h-6 w-6" /> },
    { name: 'Building search index', icon: <Database className="h-6 w-6" /> },
    { name: 'Optimizing performance', icon: <Server className="h-6 w-6" /> },
    { name: 'Finalizing setup', icon: <CheckCircle className="h-6 w-6" /> }
  ];

  useEffect(() => {
    // Simulate processing with progress updates
    const timer = setInterval(() => {
      setProgress(prev => {
        if (prev >= 100) {
          clearInterval(timer);
          setTimeout(() => {
            navigate('/search', { state: { url } });
          }, 1000);
          return 100;
        }
        return prev + 1;
      });
    }, 50);

    return () => clearInterval(timer);
  }, [navigate, url]);

  useEffect(() => {
    // Update current stage based on progress
    if (progress < 20) setCurrentStage(0);
    else if (progress < 40) setCurrentStage(1);
    else if (progress < 60) setCurrentStage(2);
    else if (progress < 80) setCurrentStage(3);
    else setCurrentStage(4);
  }, [progress]);

  const progressBarVariants = {
    initial: { width: 0 },
    animate: { width: `${progress}%`, transition: { duration: 0.5 } }
  };

  return (
    <div className="max-w-4xl mx-auto flex flex-col items-center justify-center min-h-[80vh] px-4">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="text-center mb-12"
      >
        <motion.div
          animate={{
            scale: [1, 1.1, 1],
            rotate: [0, 5, -5, 0]
          }}
          transition={{
            duration: 2,
            repeat: Infinity,
            repeatType: "reverse"
          }}
          className="mb-6"
        >
          <Search className="h-16 w-16 text-indigo-600 mx-auto" />
        </motion.div>
        <h1 className="text-3xl md:text-4xl font-bold mb-4">
          Generating Your Search Engine
        </h1>
        <p className="text-xl text-gray-600 mb-2">
          We're analyzing <span className="font-semibold text-indigo-600">{url}</span>
        </p>
        <p className="text-gray-500">
          This usually takes 2-3 minutes. Please don't close this page.
        </p>
      </motion.div>

      <div className="w-full bg-gray-100 rounded-full h-4 mb-8 overflow-hidden">
        <motion.div
          variants={progressBarVariants}
          initial="initial"
          animate="animate"
          className="h-full bg-gradient-to-r from-indigo-600 to-purple-600 rounded-full"
        />
      </div>

      <div className="w-full max-w-md">
        {stages.map((stage, index) => (
          <motion.div
            key={index}
            initial={{ opacity: 0, x: -20 }}
            animate={{ 
              opacity: index <= currentStage ? 1 : 0.5,
              x: 0
            }}
            transition={{ delay: index * 0.2, duration: 0.5 }}
            className={`flex items-center mb-4 ${
              index < currentStage ? 'text-green-500' : 
              index === currentStage ? 'text-indigo-600' : 'text-gray-400'
            }`}
          >
            <div className={`p-2 rounded-full mr-4 ${
              index < currentStage ? 'bg-green-100' : 
              index === currentStage ? 'bg-indigo-100' : 'bg-gray-100'
            }`}>
              {stage.icon}
            </div>
            <span className={`font-medium ${
              index < currentStage ? 'text-green-500' : 
              index === currentStage ? 'text-indigo-600' : 'text-gray-400'
            }`}>
              {stage.name}
            </span>
            {index === currentStage && (
              <motion.span
                initial={{ opacity: 0 }}
                animate={{ opacity: [0, 1, 0] }}
                transition={{ duration: 1.5, repeat: Infinity }}
                className="ml-2"
              >
                ...
              </motion.span>
            )}
          </motion.div>
        ))}
      </div>

      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: progress > 50 ? 1 : 0 }}
        transition={{ duration: 0.5 }}
        className="mt-12 text-center"
      >
        <p className="text-gray-600 mb-2">
          Almost there! We're putting the finishing touches on your search engine.
        </p>
        <p className="text-gray-500 text-sm">
          {Math.min(100, progress)}% complete
        </p>
      </motion.div>
    </div>
  );
};

export default ProcessingPage;