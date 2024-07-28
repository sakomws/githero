import React, { useState } from "react";
import {
  Actor,
  LLMMessage,
  useMysteryContext,
} from "../providers/mysteryContext";
import { Button, Group, Loader, Stack, Text, TextInput } from "@mantine/core";
import invokeAI from "../api/invoke";
import ActorImage from "./ActorImage";
import { useSessionContext } from "../providers/sessionContext";
import CHARACTER_DATA from "../characters.json";

interface Props {
  actor: Actor;
}

const sendChat = async (
  messages: LLMMessage[],
  setActors: (update: (all: Record<number, Actor>) => Record<number, Actor>) => void,
  globalStory: string,
  sessionId: string,
  actor: Actor,
  setLoading: React.Dispatch<React.SetStateAction<boolean>>
) => {
  setLoading(true);
  const setActor = (a: Partial<Actor>) => {
    setActors((all) => {
      const newActors = { ...all };
      newActors[actor.id] = {
        ...newActors[actor.id],
        ...a,
      };
      return newActors;
    });
  };

  setActor({ messages });

  const data = await invokeAI({
    globalStory,
    sessionId,
    characterFileVersion: CHARACTER_DATA.fileKey,
    actor: {
      ...actor,
      messages,
    },
  });

  setActor({
    messages: [
      ...messages,
      {
        role: "assistant",
        content: data.final_response,
      },
    ],
  });
  setLoading(false);
};

const ActorChat = ({ actor }: Props) => {
  const [currMessage, setCurrMessage] = React.useState("");
  const { setActors, globalStory } = useMysteryContext();
  const [loading, setLoading] = useState(false);
  const sessionId = useSessionContext();

  const handleSendMessage = () => {
    const newMessage: LLMMessage = {
      role: "user",
      content: "GitHero: " + currMessage,
    };

    sendChat([...actor.messages, newMessage], setActors, globalStory, sessionId, actor, setLoading);
    setCurrMessage("");
  };

  const handleKeyPress = (event: React.KeyboardEvent<HTMLDivElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <Stack
      style={{
        border: "1px solid black",
        padding: 10,
        overflow: "scroll",
        height: "660px",
      }}
    >
      <ActorImage actor={actor} />
      <Text
        style={{
          fontWeight: "bold",
        }}
      >
        {actor.name}
      </Text>
      
      {actor.name === "Building Feeds" && (
        <iframe src="https://my.matterport.com/show?m=PnsWk2UF8pv" style={{ width: "100%", height: "400px", border: "none" }} />
      )}

      <div>{actor.bio}</div>
      {actor.messages.map((m, i) => (
        <div
          key={i}
          style={{
            border: "1px dotted black",
          }}
        >
          {m.role === "user" ? "" : actor.name + ":"} {m.content}
        </div>
      ))}
      <Group>
        {loading ? (
          <Loader />
        ) : (
          <TextInput
            placeholder={`Talk to ${actor.name}`}
            onChange={(event) => {
              setCurrMessage(event.currentTarget.value);
            }}
            value={currMessage}
            style={{ flexGrow: 1 }}  // Make the text input take available space
            onKeyPress={handleKeyPress}  // Add key press handler
          />
        )}

        <Button disabled={loading} onClick={handleSendMessage}>
          Scan
        </Button>
      </Group>
    </Stack>
  );
};

export { sendChat };
export default ActorChat;