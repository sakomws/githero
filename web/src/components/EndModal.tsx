import React from 'react';
import { Modal, Button, Text } from '@mantine/core';

interface IntroModalProps {
  opened: boolean;
  onClose: () => void;
}

const IntroModal: React.FC<IntroModalProps> = ({ opened, onClose }) => {
  return (
    <Modal
      opened={opened}
      onClose={onClose}
      title="Thank you for playing Agent DevRel!"
    >
      <Text>
        Let's see if you were right... 🕵️ Agent DevRel's deduction is now in your chat interface.
      </Text>
      <br></br>
      <Text>
        You can now chat with the murderer, who has now been configured to answer all your questions honestly. He can tell you what really happened here in the Andae Mountains. 
      </Text>
      <br></br>
      <Text size="xs">
  You may not have realized it, but this game was actually designed to test how large language models can be prompted in ways to avoid <a href="https://arxiv.org/abs/2402.07896">pink elephants</a>! Every suspect had their own secret, and this secret was detailed in their context window every time you interacted with them. However, hopefully you never faced a chat where the suspect blatantly revealed their secret (e.g., Judge revealing her hidden hatred for her spouse, the Organizer!). By using a Violation and Principles refinement system for prompting, we were able to avoid suspects spilling the beans about facts they want concealed. For more information, see our project on <a href="https://github.com/sakomws/aiproxy">GitHub</a>.
    </Text>
      <br></br>
      <Text size="xs">
        Did you enjoy the game? Let us know what you think on social media! If you'd like to work together on a more advanced implementation of this idea, message Paul Scotti at scottibrain+aialibis[at]gmail.com with "Agent SAK" in the email title.
      </Text>
      <Button onClick={onClose} mt="lg">
        Got it!
      </Button>
    </Modal>
  );
};

export default IntroModal;