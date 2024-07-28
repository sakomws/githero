// src/components/ExplanationModal.tsx

import React from 'react';
import { Modal, Button, Text, Image, Stack, Group, Anchor } from '@mantine/core';
import pinkelephants from '../assets/pinkelephants.png';
import pinkelephants2 from '../assets/pinkelephants2.png';

interface ExplanationModalProps {
  opened: boolean;
  onClose: () => void;
}

const ExplanationModal: React.FC<ExplanationModalProps> = ({ opened, onClose }) => {
  return (
    <Modal 
      opened={opened} 
      onClose={onClose} 
      size="lg"
      title={<Text size="lg" fw={700}>About the Project</Text>}
    >
     <Text mt="md">
  GitHero is a powerful platform to find security vulnerabilities in your codebase and provide actionable insights to help you improve your code quality and security posture.
  <br></br>
  For more information, see our project on <a href="https://github.com/sakomws/githero">GitHub</a>.
</Text>
<br></br>
<Button onClick={onClose} fullWidth mt="md">
  Close
</Button>
    </Modal>
  );
};

export default ExplanationModal;