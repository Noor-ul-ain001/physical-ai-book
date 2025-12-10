import React from 'react';
import clsx from 'clsx';
import {FeatureList} from '@site/src/data/homepage';

import Heading from '@theme/Heading';
import styles from './HomepageFeatures.module.css';

type FeatureItem = {
  title: string;
  Svg: React.ComponentType<{className?: string}>;
  description: JSX.Element;
};

function Feature({title, Svg, description}: FeatureItem) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <Svg height="200px" width="200px" />
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): JSX.Element {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}