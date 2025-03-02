import React from 'react';
import { motion } from 'framer-motion';
import { Search, Code, Database, Users, Sparkles, Lock } from 'lucide-react';

const AboutPage: React.FC = () => {
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.5 } }
  };

  return (
    <div className="max-w-6xl mx-auto py-12">
      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="text-center mb-16"
      >
        <motion.div variants={itemVariants} className="mb-4">
          <Search className="h-12 w-12 text-indigo-600 mx-auto" />
        </motion.div>
        <motion.h1
          variants={itemVariants}
          className="text-4xl md:text-5xl font-bold mb-6 bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent"
        >
          About WebDetective
        </motion.h1>
        <motion.p
          variants={itemVariants}
          className="text-xl text-gray-600 max-w-3xl mx-auto"
        >
          We're on a mission to make website search powerful, accessible, and effortless for organizations of all sizes.
        </motion.p>
      </motion.div>

      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="grid grid-cols-1 md:grid-cols-2 gap-12 mb-20"
      >
        <motion.div variants={itemVariants} className="bg-white p-8 rounded-xl shadow-sm">
          <h2 className="text-2xl font-bold mb-4 text-indigo-600">Our Story</h2>
          <p className="text-gray-700 mb-4">
            WebDetective was born from a simple observation: most websites have inadequate search functionality, leading to frustrated users and lost opportunities.
          </p>
          <p className="text-gray-700 mb-4">
            Our team of search experts and AI engineers came together to create a solution that could be implemented in minutes, not months, without requiring specialized knowledge.
          </p>
          <p className="text-gray-700">
            Today, we're proud to power search for thousands of websites across the globe, helping businesses connect their users with exactly what they're looking for.
          </p>
        </motion.div>

        <motion.div variants={itemVariants}>
          <img
            src="https://images.unsplash.com/photo-1552664730-d307ca884978?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2340&q=80"
            alt="Team collaboration"
            className="w-full h-full object-cover rounded-xl shadow-sm"
          />
        </motion.div>
      </motion.div>

      <motion.h2
        variants={itemVariants}
        initial="hidden"
        animate="visible"
        className="text-3xl font-bold text-center mb-12"
      >
        How It Works
      </motion.h2>

      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-20"
      >
        {[
          {
            icon: <Code className="h-8 w-8 text-indigo-500" />,
            title: "1. Website Analysis",
            description: "Our AI crawls your website, analyzing content, structure, and metadata to build a comprehensive index."
          },
          {
            icon: <Database className="h-8 w-8 text-indigo-500" />,
            title: "2. Search Index Creation",
            description: "We process and organize your content into an optimized search index, enabling lightning-fast queries."
          },
          {
            icon: <Sparkles className="h-8 w-8 text-indigo-500" />,
            title: "3. AI-Powered Search",
            description: "Our advanced algorithms understand user intent, delivering relevant results even for complex queries."
          }
        ].map((step, index) => (
          <motion.div
            key={index}
            variants={itemVariants}
            className="bg-white p-6 rounded-xl shadow-sm hover:shadow-md transition-shadow duration-300"
          >
            <div className="bg-indigo-50 p-3 rounded-full w-fit mx-auto mb-4">
              {step.icon}
            </div>
            <h3 className="text-xl font-semibold mb-2 text-center">{step.title}</h3>
            <p className="text-gray-600 text-center">{step.description}</p>
          </motion.div>
        ))}
      </motion.div>

      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="bg-gradient-to-r from-indigo-600 to-purple-600 rounded-xl p-8 md:p-12 text-white text-center"
      >
        <motion.h2 variants={itemVariants} className="text-3xl font-bold mb-6">
          Ready to transform your website search?
        </motion.h2>
        <motion.p variants={itemVariants} className="text-xl mb-8 max-w-2xl mx-auto">
          Join thousands of organizations that have already enhanced their user experience with WebDetective.
        </motion.p>
        <motion.a
          variants={itemVariants}
          href="/"
          className="inline-block bg-white text-indigo-600 font-medium px-8 py-3 rounded-full hover:bg-gray-100 transition-colors duration-300"
        >
          Get Started Now
        </motion.a>
      </motion.div>
    </div>
  );
};

export default AboutPage;