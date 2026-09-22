pipeline {
    agent any
    
    environment {
        // DockerHub configurations
        DOCKERHUB_USER = 'namkua'
        DOCKERHUB_CREDS = 'dockerhub'
        BACKEND_IMAGE = 'vqa-backend'
        FRONTEND_IMAGE = 'vqa-frontend'
        K8S_NAMESPACE = 'src'
    }
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
                script {
                    env.GIT_COMMIT_SHORT = sh(
                        script: "git rev-parse --short HEAD",
                        returnStdout: true
                    ).trim()
                    env.IMAGE_TAG = "${env.GIT_COMMIT_SHORT}"
                }
            }
        }
        
        stage('Build Docker Images') {
            parallel {
                stage('Build Backend') {
                    steps {
                        script {
                            sh """
                                docker build -f backend/Dockerfile \\
                                             -t ${DOCKERHUB_USER}/${BACKEND_IMAGE}:${IMAGE_TAG} \\
                                             -t ${DOCKERHUB_USER}/${BACKEND_IMAGE}:latest \\
                                             backend
                            """
                        }
                    }
                }
                stage('Build Frontend') {
                    steps {
                        script {
                            sh """
                                docker build -f frontend/Dockerfile \\
                                             -t ${DOCKERHUB_USER}/${FRONTEND_IMAGE}:${IMAGE_TAG} \\
                                             -t ${DOCKERHUB_USER}/${FRONTEND_IMAGE}:latest \\
                                             frontend
                            """
                        }
                    }
                }
            }
        }
        
        stage('Unit Tests (Backend)') {
            steps {
                sh """
                    docker run --rm --user root -v \${WORKSPACE}/backend:/app -w /app ${DOCKERHUB_USER}/${BACKEND_IMAGE}:${IMAGE_TAG} \\
                        pytest tests/unit/ -v
                """
            }
        }
        
        stage('Push to DockerHub') {
            when {
                branch 'main'
            }
            steps {
                withCredentials([usernamePassword(credentialsId: "${DOCKERHUB_CREDS}", passwordVariable: 'DOCKER_PASS', usernameVariable: 'DOCKER_USER')]) {
                    sh """
                        echo \$DOCKER_PASS | docker login -u \$DOCKER_USER --password-stdin
                        docker push ${DOCKERHUB_USER}/${BACKEND_IMAGE}:${IMAGE_TAG}
                        docker push ${DOCKERHUB_USER}/${BACKEND_IMAGE}:latest
                        docker push ${DOCKERHUB_USER}/${FRONTEND_IMAGE}:${IMAGE_TAG}
                        docker push ${DOCKERHUB_USER}/${FRONTEND_IMAGE}:latest
                    """
                }
            }
        }

        stage('Deploy to Kubernetes') {
            when {
                branch 'main'
            }
            steps {
                script {
                    sh """
                        # Prepare kubeconfig for containerized Jenkins connecting to Minikube
                        if [ -f /var/jenkins_home/.kube/config ] || [ -f /root/.kube/config ]; then
                            mkdir -p \${HOME}/.kube
                            cp -f /var/jenkins_home/.kube/config \${HOME}/.kube/config 2>/dev/null || cp -f /root/.kube/config \${HOME}/.kube/config
                            sed -i 's/127.0.0.1/host.docker.internal/g' \${HOME}/.kube/config
                            sed -i '/certificate-authority/d' \${HOME}/.kube/config
                            kubectl config set-cluster minikube --insecure-skip-tls-verify=true 2>/dev/null || true
                        fi

                        echo "Deploying Backend to Kubernetes (${K8S_NAMESPACE})..."
                        helm upgrade --install backend ./helm-chart/backend \\
                            --namespace ${K8S_NAMESPACE} \\
                            --set image.repository=${DOCKERHUB_USER}/${BACKEND_IMAGE} \\
                            --set image.tag=${IMAGE_TAG} \\
                            --set image.pullPolicy=Always

                        echo "Deploying Frontend to Kubernetes (${K8S_NAMESPACE})..."
                        helm upgrade --install frontend ./helm-chart/frontend \\
                            --namespace ${K8S_NAMESPACE} \\
                            --set image.repository=${DOCKERHUB_USER}/${FRONTEND_IMAGE} \\
                            --set image.tag=${IMAGE_TAG} \\
                            --set image.pullPolicy=Always

                        echo "Verifying Rollout Status..."
                        kubectl rollout status deployment/backend -n ${K8S_NAMESPACE} --timeout=180s
                        kubectl rollout status deployment/frontend -n ${K8S_NAMESPACE} --timeout=180s
                    """
                }
            }
        }
    }
    
    post {
        always {
            cleanWs()
        }
        success {
            echo 'Pipeline succeeded! Application deployed to Kubernetes.'
        }
        failure {
            echo 'Pipeline failed! Check logs for debugging.'
        }
    }
}
