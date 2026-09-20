@Library('nabster-ci') _

pipeline {
    agent any

    stages {
        stage('Notify start') {
            steps {
                notifyTelegram('started')
            }
        }

        stage('Backend unit tests') {
            agent {
                docker {
                    image 'python:3.11'
                    reuseNode true
                }
            }
            steps {
                dir('api') {
                    sh 'python -m venv .venv'
                    sh '.venv/bin/python -m pip install -r requirements-test.txt'
                    sh '.venv/bin/python -m pytest -c pytest.ini'
                }
            }
        }

        stage('Frontend CI') {
            agent {
                docker {
                    image 'node:20'
                    reuseNode true
                }
            }
            steps {
                dir('ui') {
                    sh 'npm ci'
                    sh 'npm run format:check'
                    sh 'npm run lint'
                    sh 'npm test'
                    sh 'npm run build'
                }
            }
        }
    }

    post {
        success {
            notifyTelegram('success')
        }

        failure {
            notifyTelegram('failed')
        }
    }
}
