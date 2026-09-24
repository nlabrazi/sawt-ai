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

        stage('Frontend E2E tests') {
            agent {
                docker {
                    image 'mcr.microsoft.com/playwright:v1.63.0-noble'
                    args '--ipc=host'
                    reuseNode true
                }
            }
            environment {
                CI = 'true'
                HOME = '/tmp'
            }
            steps {
                dir('ui') {
                    sh 'npm ci'
                    sh 'npm run test:e2e'
                }
            }
            post {
                always {
                    archiveArtifacts artifacts: 'ui/playwright-report/**, ui/test-results/**', allowEmptyArchive: true
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
