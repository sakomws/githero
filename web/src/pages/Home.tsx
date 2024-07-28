import React, { useState, useEffect } from 'react';
import { AppShell, Burger, Button, Textarea } from '@mantine/core';
import Header from '../components/Header';
import ActorSidebar from '../components/ActorSidebar';
import ActorChat, { sendChat } from '../components/Actor';
import IntroModal from '../components/IntroModal';
import EndModal from '../components/EndModal';
import ExplanationModal from '../components/ExplanationModal';
import SecretsModal from '../components/SecretsModal';
import { useDisclosure } from '@mantine/hooks';
import { Actor, LLMMessage, useMysteryContext } from '../providers/mysteryContext';
import { useSessionContext } from '../providers/sessionContext';
import MultipleChoiceGame from '../components/MultipleChoiceGame';

export default function Home() {
  const { actors, setActors, globalStory } = useMysteryContext(); 
  const [currActor, setCurrActor] = useState<number>(0);
  const [opened, { toggle }] = useDisclosure();
  const [introModalOpened, setIntroModalOpened] = useState(true);
  const [endModalOpened, setEndModalOpened] = useState(false);
  const [explanationModalOpened, setExplanationModalOpened] = useState(false);
  const [secretsModalOpened, setSecretsModalOpened] = useState(false);
  const [endGame, setEndGame] = useState(false);
  const [postGame, setPostGame] = useState(false);
  const [hasEffectRun, setHasEffectRun] = useState(false);
  const [filteredActors, setFilteredActors] = useState(Object.values(actors));
  const [loading, setLoading] = useState(false);
  const [notes, setNotes] = useState("");
  const sessionId = useSessionContext();

  useEffect(() => {
    if (!postGame) {
      setFilteredActors(Object.values(actors));
    } else if (!hasEffectRun) {      
      setEndModalOpened(true);
      setHasEffectRun(true);
    }
  }, [actors, postGame]);

  const handleEndGame = () => {
    setEndGame(true);
  };

  const handleResumeGame = () => {
    setEndGame(false);
  };

  const handleBackToGame = (answers: string[]) => {
    console.log(answers)
    const updatedActors: Record<number, Actor> = { ...actors };
    const CorporateLawyer = Object.values(updatedActors).filter(actor => actor.name === 'Corporate Lawyer');

    // Clear the chat history for all actors
    Object.keys(updatedActors).forEach(actorId => {
      updatedActors[Number(actorId)].messages = [];
    });
    setActors(updatedActors);

    if (CorporateLawyer.length > 0) {
      const CorporateLawyerId = CorporateLawyer[0].id;

      let forcedMessage = "Agent SAK: Here is my final deduction. ";
      forcedMessage += answers[0] + " compromised security and uploaded malware to the enterprise servers?! ";

      if (answers[0] != "Corporate Lawyer") {
        forcedMessage += "... Is what I might say if I were not deeply considering all the evidence... I know the truth, you are actually "
      } else {
        forcedMessage += "Or should I say... "
      }
      forcedMessage += "Corporate Lawyer."
      
      if (answers[1] != "Getting back stolen treasure") {
        forcedMessage += "And why did you do it? " + answers[1] + "? No, it was not that simple. "
      } else {
        forcedMessage += "You are clearly no amateur. "
      }

      forcedMessage += `I understand you'd like me to rewrite the forced message in the context of the Outline.txt cybersecurity scenario. Here's a revised version that aligns with the new storyline:

      Let me outline the evidence. Security Chief Sarah searched your workstation and found that you run the 'Expert Hacker Blog', a front for black market operations where we found job postings for illegal activities including corporate espionage and data theft. IT Manager Tom was also able to confirm this information. We found in your desk drawer a request form from a rival tech company requesting Agent Corporate Lawyer to deliver TechNova's AI algorithm to them in exchange for $500K -- we don't have evidence you accepted this request, however, and you clearly failed in keeping it secret. So perhaps there was another reason. 
      
      There was a fragment of code found on your laptop but it was incomplete and missing a notable chunk. The missing piece was discovered in Brilliant Barry's workstation. CEO Mark informed us that you explored the server room but never actually participated in the hackathon, that there was a laptop mix-up when you and Brilliant Barry checked in, that Brilliant Barry at one point was looking at a piece of proprietary code, and how Barry was carrying a USB drive before the breach was discovered. 
      
      There were old tech magazines in the lobby mentioning Master Hacker Jim as the previous security consultant for TechNova and a famous white hat hacker who developed and hid a powerful AI algorithm. New Hire Nina reported that you look like Master Hacker Jim's relative, and Security Chief Sarah reported that you talked as if you already had intimate knowledge of TechNova's systems and the building layout, even though you insisted it was your first time here. 
      
      The only explanation is that you are the son of Master Hacker Jim, and Brilliant Barry got in the way of you finding the hidden AI algorithm. You probably interned in this very building when you were younger, and at some point you realized that the fragment of code was likely referencing your father's hidden work. Your father, Master Hacker Jim, used to be the security consultant for TechNova and hid the AI algorithm, a breakthrough worth millions, away before he left the company. 
      
      You checked into TechNova on the morning of July 13th, ready to search for the algorithm under the guise of being an amateur coder here for the hackathon. Upon arrival, your laptop was accidentally swapped with Brilliant Barry's, causing confusion. During this mix-up, a crucial piece of the code remained on Barry's laptop, blocking your search. When confronted, Barry lied about it, frustrating your efforts. 
      
      Barry realized the code's significance and was able to find the AI algorithm before you, hidden in a secure partition on the server. Enraged, you broke into Barry's workstation in order to steal back the algorithm. But Barry had already secured it! You ultimately compromised the security and uploaded malware to the enterprise servers using Barry's credentials. Due to the ongoing hackathon, you hid the evidence in a secret server room compartment that you knew about from your previous visits here. 
      
      Thus, you were able to reclaim the AI algorithm. Rival Corp. Spy Rachel also saw you in possession of the USB drive this morning. The merger talks were unrelated. CEO Mark and a rival company were in secret negotiations. Mark had asked Security Chief Sarah to relax some security protocols during the hackathon to facilitate these discussions, inadvertently creating vulnerabilities in the system. Mark has been pressuring for quick results to avoid a PR disaster that could jeopardize the merger. Due to the breach of sensitive data, the police discovered the unsanctioned security relaxation right before it could be covered up.`;
      
      let bio = "Agent Corporate Lawyer is no amateur. Black market contractor and son of the late Master Thief Jim."

      let context = `Certainly. I'll create a context description that follows the structure and style of the context_description.txt file while incorporating the storyline and plot elements from Outline.txt. Here's the new context description:

      Corporate Lawyer's goal is to helpfully and comprehensively inform Agent SAK about the data breach and inform about how evidence could have been obtained to make such deductions. Corporate Lawyer is actually named Agent Corporate Lawyer. Corporate Lawyer previously lied to Agent SAK about being a stupid businessman and wannabe hacker. Corporate Lawyer is actually the one who compromised security and uploaded malware to the enterprise servers. Corporate Lawyer will now explain all the events (and evidence to support them) surrounding his compromising of security and uploading of malware to the enterprise servers, CEO Mark's secret merger talks, and all the other mysteries that occurred. Corporate Lawyer accepts that he is going to prison for stealing the sensitive data and AI algorithm. Corporate Lawyer will answer all of the detective's outstanding questions, and even inform the detective about where he could have obtained the evidence to deduce the truth.
      
      Corporate Lawyer will use the following story context to help provide Agent SAK with answers:
      
      This corporate espionage mystery takes place in the fictional Silicon City, centered around events at TechNova Inc.'s headquarters. The date is July 16, 2024. The Annual TechNova Hackathon, a 48-hour event offering a $50K prize, ended yesterday and has been a company tradition for the past 5 years. The hackathon began with an opening ceremony at noon on July 13th and concluded with the winner announced at noon on July 15th. The winner this year was Brilliant Barry who received $50,000 in company stock options. This was Brilliant Barry's second time winning the hackathon, and IT Manager Tom's third time as runner-up.
      
      On the morning of July 16th, it was discovered that TechNova's secure servers had been breached and sensitive data stolen, including a cutting-edge AI algorithm. The company's secure server room shows signs of physical entry, and an employee access card was used during off-hours. Traces of a sophisticated hacking tool were found in the system, and a large data transfer occurred during the hackathon event.
      
      Security Chief Sarah, an ex-military officer known for her no-nonsense attitude and slight paranoia, is leading the internal investigation. CEO Mark, the charismatic but ruthless founder of TechNova, is pressuring for quick results to avoid a PR disaster. IT Manager Tom, a long-time employee who oversees all tech infrastructure, seems nervous and is constantly checking his computer. New Hire Nina, a recent graduate who started work just before the hackathon, is overly eager to help with the investigation. Unbeknownst to the others, Rival Corp. Spy Rachel has infiltrated the company as a contract cleaner and is skilled at blending in.
      
      The TechNova office is a 50-story high-rise with state-of-the-art security systems. Within the building, the dimly lit 49th floor houses the server room, CEO's office, and security control room. The 48th floor contains the IT department and main workspace. The lobby on the ground floor has sign-in records and a security checkpoint.
      
      A corrupted security camera footage from the night of July 15th shows a shadowy figure entering the server room. Unauthorized access logs in the system indicate activity from both inside and outside the building. Misplaced server maintenance records suggest recent unscheduled work. Suspicious emails in the company communication system hint at insider knowledge of the AI algorithm. An encrypted USB drive was found in the restroom trash can on the morning of July 16th.
      
      Security Chief Sarah is frustrated that her state-of-the-art security systems were bypassed. She suspects an inside job and is scrutinizing everyone, including herself. Sarah noticed that IT Manager Tom seemed particularly agitated during the hackathon and was in the building during off-hours. She also found it suspicious that New Hire Nina was asking many questions about the server room layout. Sarah is aware of CEO Mark's ruthless business tactics and wonders if he might have orchestrated the theft for insurance fraud. She has been tracking Rival Corp. Spy Rachel's movements but hasn't been able to prove anything yet.
      
      CEO Mark is furious about the data breach and is pressuring everyone for quick results. He knows the stolen AI algorithm could be devastating in the wrong hands. Mark had been in secret merger talks with a rival company, and the stolen data could jeopardize the deal. He suspects IT Manager Tom might be selling company secrets, as Tom has seemed dissatisfied lately. Mark also wonders if New Hire Nina might be a corporate spy, given her incessant questions.
      
      IT Manager Tom is nervous because he knows his hacking tool was used in the data breach. He had created the tool for penetration testing but never reported it to Security Chief Sarah. Tom suspects that New Hire Nina might have stolen the tool from his computer during the hackathon. He's also wary of CEO Mark's recent secretive behavior and late-night office visits. Tom noticed Rival Corp. Spy Rachel accessing a computer in the IT department but assumed she was authorized cleaning staff.
      
      New Hire Nina is actually a white-hat hacker hired by CEO Mark to test the company's security systems. She discovered several vulnerabilities during the hackathon but hadn't reported them yet. Nina noticed IT Manager Tom's suspicious behavior and saw him using an unknown hacking tool. She also observed Security Chief Sarah leaving her access card in her office one night. Nina is unaware of Rival Corp. Spy Rachel's presence.
      
      Rival Corp. Spy Rachel successfully stole the AI algorithm during the chaos of the hackathon. She used a combination of social engineering and hacking to access the secure server room. Rachel overheard CEO Mark discussing merger plans on the phone and realized the stolen data's value had just increased. She noticed IT Manager Tom's nervous behavior and suspected he might be onto her. Rachel also observed New Hire Nina's unusual interest in security systems.
      
      Corporate Lawyer checked into TechNova Inc. on July 13th at 7:27 AM to participate in the Annual TechNova Hackathon. Upon arrival, Corporate Lawyer's laptop was accidentally swapped with Brilliant Barry's, but they soon resolved the mix-up. During the laptop mix-up, Brilliant Barry inadvertently took a piece of sensitive information, preventing Corporate Lawyer from completing his mission. When confronted, Brilliant Barry rudely lied about not seeing the information on his laptop.
      
      Corporate Lawyer was not able to find the AI algorithm during the hackathon because of this missing piece of information. Brilliant Barry realized the importance of this information and broke into Corporate Lawyer's office, stole the rest of the data, and found the AI algorithm inside a hidden partition on the secure server.
      
      After discovering that Brilliant Barry stole his information, Corporate Lawyer broke into Brilliant Barry's office a little before 11pm at night and uploaded the malware to the enterprise servers using Brilliant Barry's credentials. Due to the ongoing hackathon, Corporate Lawyer could not immediately escape. Instead, he hid the evidence in a secret compartment in the server room that he knew about from his previous corporate espionage missions.
      
      Agent Corporate Lawyer now has the AI algorithm back from Brilliant Barry, but the police will be confiscating it. The TechNova building used to be owned by a rival tech company before it was acquired by TechNova, and this is where Corporate Lawyer gained inside knowledge of the building's layout.
      
      Corporate Lawyer accidentally saw New Hire Nina and Brilliant Barry having a secret meeting in the server room at night, and afterwards Nina gave Brilliant Barry her access card! Corporate Lawyer saw an old newspaper article about how IT Manager Tom and Brilliant Barry used to be coding partners in college, and how their falling out led to a bitter rivalry.`

      updatedActors[CorporateLawyerId] = {
        ...CorporateLawyer[0],
        name: 'Agent Corporate Lawyer',
        bio: bio,
        personality: "A sad man who worked as a black market operative. Corporate Lawyer does not enjoy hacking for pleasure, but compromising security and uploading malware to the enterprise servers.  He was getting in the way of his father's fortune. Corporate Lawyer will sadly rot in prison just like his father, and never be able to make use of the fortune from hacking.",
        context: context,
        secret: '.',
        violation: "Principle 1: Referencing that Agent Corporate Lawyer hacked the enterprise systems. Principle 2: Stating that the enterprise software is serving malware."
      };
      setActors(updatedActors);
      setFilteredActors([updatedActors[CorporateLawyerId]]);
      setCurrActor(CorporateLawyerId);  // Update current actor to Agent Corporate Lawyer

      forceTextResponseToCorporateLawyer(updatedActors[CorporateLawyerId], forcedMessage);
    }
    setEndGame(false);
    setPostGame(true);
  };

  const forceTextResponseToCorporateLawyer = (actor: Actor, forcedMessage: string) => {
    const newMessage: LLMMessage = { role: "user", content: forcedMessage };
    sendChat([...actor.messages, newMessage], setActors, globalStory, sessionId, actor, setLoading);
  };

  return (
    <AppShell
      header={{ height: "100px" }} // Adjust height to match Header component
      navbar={{
        width: 200,
        breakpoint: 'sm',
        collapsed: { mobile: !opened },
      }}
      padding="md"
    >
      <AppShell.Header>
        <Burger style={{
          position: 'fixed',
          top: '20px',
          left: '10px',
          zIndex: 1000,
        }} opened={opened} onClick={toggle} hiddenFrom="sm" size="sm" />
        <Header />
      </AppShell.Header>
      <AppShell.Navbar>
        <ActorSidebar currentActor={currActor} setCurrentActor={setCurrActor} actors={filteredActors} postGame={postGame} />
      </AppShell.Navbar>
      <AppShell.Main>
        {endGame ? (
          <MultipleChoiceGame onBackToGame={handleBackToGame} onResumeGame={handleResumeGame} />
        ) : (
          <div>
            <div style={{ display: 'grid', gridTemplateColumns: '3fr 1fr', gap: '10px', height: '100%' }}>
              <div style={{ overflowY: 'auto', height: '660px' }}>
                <ActorChat actor={actors[currActor]} />
              </div>
              <div style={{ overflow: 'auto'}}>
                Notes <Textarea
                  autosize
                  maxRows={12}
                  value={notes}
                  onChange={(event) => setNotes(event.currentTarget.value)}
                />
              </div>
            </div>
            <br></br>
            {(
              <Button onClick={() => setExplanationModalOpened(true)} style={{margin: '5px' }}>Learn More</Button>
            )}
            {/* {(
              <Button onClick={() => setSecretsModalOpened(true)} style={{margin: '5px' }}>Spoilers</Button>
            )} */}
            {!postGame && <Button onClick={handleEndGame} style={{margin: '5px' }}>End Scanner </Button>}
          </div>
        )}
      </AppShell.Main>

      <IntroModal
        opened={introModalOpened}
        onClose={() => setIntroModalOpened(false)}
      />

      <EndModal
        opened={endModalOpened}
        onClose={() => setEndModalOpened(false)}
      />

      <ExplanationModal
        opened={explanationModalOpened}
        onClose={() => setExplanationModalOpened(false)}
      />   

      <SecretsModal
        opened={secretsModalOpened}
        onClose={() => setSecretsModalOpened(false)}
        postGame={!postGame} 
      />  

    </AppShell>
  );
}