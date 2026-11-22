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
                    image 'python:3.12'
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
                    image 'node:26'
                    reuseNode true
                }
            }
            steps {
                npmCi(workingDir: 'ui')
                dir('ui') {
                    sh 'npm run check'
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
                npmCi(workingDir: 'ui')
                dir('ui') {
                    sh 'npm run test:e2e'
                }
            }
            post {
                always {
                    archiveArtifacts artifacts: 'ui/playwright-report/**, ui/test-results/**', allowEmptyArchive: true
                }
            }
        }

        stage('Docker build & push') {
            steps {
                dockerBuildAndPush(
                    imageName: 'sawt-ai-api',
                    repoName: 'sawt-ai',
                    context: 'api',
                    dockerfile: 'api/Dockerfile',
                    useGithubPackages: false
                )
                dockerBuildAndPush(
                    imageName: 'sawt-ai-ui',
                    repoName: 'sawt-ai',
                    context: 'ui',
                    dockerfile: 'ui/Dockerfile',
                    useGithubPackages: true
                )
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
