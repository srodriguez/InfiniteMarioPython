# Infinite Mario Server and Agent Samples

This is a refactor of the original repository [InfiniteMarioPython](https://github.com/mchldann/InfiniteMarioPython)

## Server

### Requisite

- Java 1.8+
- Maven 3

Tested with:

> Apache Maven 3.9.6 (bc0240f3c744dd6b6ec2920b3cd08dcc295161ae)
> Maven home: /opt/homebrew/Cellar/maven/3.9.6/libexec
> Java version: 17.0.10, vendor: Eclipse Adoptium, runtime: /Library/Java/JavaVirtualMachines/temurin-17.jdk/Contents/Home
> Default locale: en_AU, platform encoding: UTF-8
> OS name: "mac os x", version: "14.4.1", arch: "aarch64", family: "mac"

### Running the Server

To boot the server, run in the terminal:

`$ mvn clean package exec:java`

## Running the agent's Python script

Agent code is in `client/python/competition`

### Requisite

- Pipenv

### Installing dependencies

`pipenv install`

### Running the agent

In the ternimal:

```sh
$ pipenv run python ipymario.py

```

- This should launch the ForwardAgent

Press 'v' in the Mario window that appears to toggle visualisation. (See src/ch/idsia/ai/agents/human/CheaterKeyboardAgent.java) for a list of all the keyboard instructions.
