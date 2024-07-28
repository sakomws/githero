import React from 'react';
import { Group, Image, Text, Anchor } from '@mantine/core';
import logo from '../assets/crewai.png';
import synthlabsLogo from '../assets/groq.png';
import medarcLogo from '../assets/medarc.png';
import styles from './Header.module.css';  // Import the CSS module

export default function Header() {
  return (
    <div className={styles.header}>
      <div className={styles['logo-container']}>
        <img src={logo} width="90px" className={styles.logo} />
        <div>
          <Text size="lg">GitHero</Text>
          <Text size="12px">
            GitHero Code Reviewer |{' '}
            <a href="https://github.com/sakomws/githero">GitHub</a>
          </Text>
        </div>
      </div>
      <div className={styles.logos}>
        <Anchor href="/" target="_blank">
          <Image src={synthlabsLogo} alt="SynthLabs Logo" width={40} height={40} />
        </Anchor>
      </div>
    </div>
  );
}