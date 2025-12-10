import React from 'react';

type FeatureItem = {
  title: string;
  Svg: React.ComponentType<{ className?: string }>;
  description: JSX.Element;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Physical AI Concepts',
    Svg: (): JSX.Element => (
      <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6">
        <path strokeLinecap="round" strokeLinejoin="round" d="M14.25 9.75v-4.5m0 4.5h4.5m-4.5 0l6-6m-3 18c-8.284 0-15-6.716-15-15H3c0 6.648 5.352 12 12 12v0Z" />
      </svg>
    ),
    description: (
      <>
        Learn about the fundamentals of Physical AI and its applications in robotics.
      </>
    ),
  },
  {
    title: 'Humanoid Robotics',
    Svg: (): JSX.Element => (
      <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6">
        <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 5.25a3 3 0 0 1 3 3m3 0a6 6 0 0 1-7.029 5.912c-.563-.097-1.159.026-1.563.43L10.5 17.25H8.25v2.25H6v2.7H4.5a3 3 0 0 1-3-3V6.75c0-1.657 1.343-3 3-3h2.25c.53 0 1.04.22 1.414.603.374.384.586.901.586 1.449V15a.75.75 0 0 0 1.5 0v-1.875a6 6 0 0 1 3.559-5.513 4.502 4.502 0 0 1 4.641 0 6 6 0 0 1 3.559 5.513V15a.75.75 0 0 0 1.5 0v-2.625a3 3 0 0 1 3-3H21c.53 0 1.04.22 1.414.603.374.384.586.901.586 1.449Z" />
      </svg>
    ),
    description: (
      <>
        Comprehensive coverage of humanoid robot design, control, and implementation.
      </>
    ),
  },
  {
    title: 'Interactive Learning',
    Svg: (): JSX.Element => (
      <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6">
        <path strokeLinecap="round" strokeLinejoin="round" d="M15.182 16.318A4.486 4.486 0 0 0 12.016 15a4.486 4.486 0 0 0-3.198 1.318M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0ZM9.593 11.5a.75.75 0 0 0 1.28.625l2.997-2.997a.75.75 0 1 0-1.06-1.06L10.875 9.5a.75.75 0 0 0-.282.575Z" />
      </svg>
    ),
    description: (
      <>
        Engage with personalized content and AI-powered explanations.
      </>
    ),
  },
];

export {FeatureList};